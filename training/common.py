"""Shared pieces of the Python training pipeline (training only — inference runs in ../rust).

Contract with the Rust side (rust/src/encoder.rs, rust/src/preprocess.rs):
  * preprocessing: preprocess/preprocessor.py (the Rust port is bit-exact to it) → 138×138 gray in [0,1]
  * encoder input  "input"     [b, 3, 138, 138] float32, gray replicated to 3 channels, no mean/std
  * encoder output "embedding" [b, 512] float32 (ResNet-18 global-average-pooled features)
  * identity of an image named <session>_<number>: (number − 1) // block (10 images per palm per session)
"""
import json, os, re, sys, time

import numpy as np

SIDE = 138
EMBED_DIM = 512
NAME_RE = re.compile(r"^(?P<session>.+)_(?P<num>\d+)$")


def log(event, **kw):
    """One JSON line per event, so a notebook or CI can follow progress."""
    print(json.dumps({"t": round(time.time(), 1), "event": event, **kw}), flush=True)


def identity(name, block=10):
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"image name {name!r} is not <session>_<number>")
    return m["session"], (int(m["num"]) - 1) // block


def split_ids(ids, seed=0, frac=(0.70, 0.15, 0.15)):
    """Disjoint identity split → {'train','cal','test'}: open-set (test palms are never seen in training)."""
    u = np.array(sorted(set(int(i) for i in ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    a, b = int(round(frac[0] * len(u))), int(round((frac[0] + frac[1]) * len(u)))
    return {"train": sorted(u[:a].tolist()), "cal": sorted(u[a:b].tolist()), "test": sorted(u[b:].tolist())}


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def import_preprocessor():
    sys.path.insert(0, repo_root())
    from preprocess.preprocessor import preprocess_image  # the exact code the Rust port matches
    return preprocess_image
