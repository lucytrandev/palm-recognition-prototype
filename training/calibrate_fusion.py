"""Choose how to fuse the CNN embedding with palm-ridge codes (Python side of the split).

    palmprint pair-scores --split cal --out data/cal_pairs.npy        # Rust: features → pair scores
    python training/calibrate_fusion.py data/cal_pairs.npy rust/models/palmprint_encoder_512.card.json

fused = w · cosine + (1 − w) · (1 − ridge_distance), computed in float32 in the same order as
rust/src/ridges.rs::fuse, so the threshold means the same thing on both sides. w is the grid value
with the lowest EER on the calibration palms (w = 1 is CNN only, w = 0 ridges only); the threshold
follows the same FAR rule as train.py (lean/Match.lean: far_threshold_bound). The result is written
into the model card's "fusion" entry, which every Rust command reads; `palmprint eval` then reports
CNN, ridge and fused MCC on the unseen test palms.
"""
import argparse, json

import numpy as np

from common import log
from metrics import far_threshold, mcc


def eer(scores, genuine):
    gs, ims = np.sort(scores[genuine]), np.sort(scores[~genuine])
    thr = np.unique(scores)
    far = 1 - np.searchsorted(ims, thr, side="left") / max(len(ims), 1)
    frr = np.searchsorted(gs, thr, side="left") / max(len(gs), 1)
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2)


def fuse(cos, ridge, w):
    w = np.float32(w)
    return (w * cos + (np.float32(1) - w) * (np.float32(1) - ridge)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", help="pair-scores output: rows [cosine, ridge_distance, genuine]")
    ap.add_argument("card", help="model card to update")
    ap.add_argument("--target-far", type=float, default=None, help="default: the card's target FAR")
    ap.add_argument("--ridge-shift", type=int, default=1, help="must match the pair-scores run")
    ap.add_argument("--step", type=float, default=0.05)
    a = ap.parse_args()
    p = np.load(a.pairs).astype(np.float32)
    cos, ridge, genuine = p[:, 0], p[:, 1], p[:, 2] > 0.5
    card = json.load(open(a.card))
    alpha = a.target_far if a.target_far is not None else card.get("target_far", 1e-3)
    grid = np.round(np.arange(0, 1 + 1e-9, a.step), 4)
    curve = [(float(w), eer(fuse(cos, ridge, w), genuine)) for w in grid]
    w, best = min(curve, key=lambda x: (x[1], -x[0]))  # ties → more CNN weight
    s = fuse(cos, ridge, w)
    t = far_threshold(s[~genuine], alpha)
    tp, fp = int((s[genuine] >= t).sum()), int((s[~genuine] >= t).sum())
    fn, tn = int(genuine.sum()) - tp, int((~genuine).sum()) - fp
    card["fusion"] = {
        "weight": w, "threshold": t, "ridge_shift": a.ridge_shift, "target_far": alpha,
        "calibration": {"pairs": int(len(s)), "genuine": int(genuine.sum()), "eer": best,
                        "eer_cnn_only": dict(curve)[1.0], "eer_ridges_only": dict(curve)[0.0],
                        "at_threshold": {"tar": tp / max(tp + fn, 1), "far": fp / max(fp + tn, 1), "mcc": mcc(tp, fp, fn, tn)},
                        "eer_by_weight": curve},
    }
    json.dump(card, open(a.card, "w"), indent=1)
    log("fusion", weight=w, threshold=t, cal_eer=best, cal_eer_cnn=dict(curve)[1.0], cal_eer_ridges=dict(curve)[0.0],
        cal_mcc=card["fusion"]["calibration"]["at_threshold"]["mcc"])


if __name__ == "__main__":
    main()
