"""Raw palm images → one uint8 ROI archive with identity labels.

    python training/prepare.py dataset/archive/session1 dataset/archive/session2 -o data/rois.npz

Uses preprocess/preprocessor.py, the code the Rust inference port reproduces bit-exactly, so training
sees exactly what inference sees. ROIs are stored as uint8: the normalised ROI is resized_uint8 / 255,
so this is lossless. Also accepts an existing folder of preprocessed .npy files (batch_preprocess.py).
"""
import argparse, os

import numpy as np

from common import SIDE, identity, import_preprocessor, log

EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="image folders (folder name = session) or one folder of <session>_<n>.npy ROIs")
    ap.add_argument("-o", "--out", default="data/rois.npz")
    ap.add_argument("--block", type=int, default=10, help="consecutive images per palm per session")
    a = ap.parse_args()
    pre = import_preprocessor()
    rois, names, failed = [], [], 0
    for d in a.dirs:
        folder = os.path.basename(os.path.normpath(d))
        files = sorted(os.listdir(d))
        for i, f in enumerate(files):
            stem, ext = os.path.splitext(f)
            p = os.path.join(d, f)
            if ext.lower() == ".npy":
                r, name = np.load(p), stem
            elif ext.lower() in EXT:
                r, name = pre(p), f"{folder}_{stem}"
            else:
                continue
            if r is None or r.shape != (SIDE, SIDE):
                failed += 1
                continue
            rois.append(np.round(r * 255.0).astype(np.uint8))
            names.append(name)
            if len(names) % 500 == 0:
                log("prepare", done=len(names), failed=failed, folder=folder, of=len(files))
    sess, ids = zip(*(identity(n, a.block) for n in names))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, rois=np.stack(rois), names=np.array(names), ids=np.array(ids, np.int64), sessions=np.array(sess))
    log("prepared", images=len(names), failed=failed, palms=len(set(ids)), sessions=sorted(set(sess)), out=a.out)


if __name__ == "__main__":
    main()
