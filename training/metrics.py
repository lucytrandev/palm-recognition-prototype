"""Verification / identification metrics, same definitions as `palmprint eval` (rust/src/eval.rs).

All pairs i<j of the given images: genuine if same palm. Accept ⇔ cosine ≥ threshold.
MCC is the accept/reject decision over all pairs (None when undefined).
"""
import math

import numpy as np


def mcc(tp, fp, fn, tn):
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return None if d == 0 else (tp * tn - fp * fn) / d


def far_threshold(impostor, alpha):
    """Smallest float32 strictly above the (⌊α·N⌋+1)-th largest impostor score: at most ⌊α·N⌋ impostor
    pairs reach it, so the calibration FAR is ≤ α (proved in lean/Match.lean, far_threshold_bound)."""
    s = np.sort(np.asarray(impostor, np.float32))[::-1]
    k = int(math.floor(alpha * len(s)))
    if k >= len(s):
        return float(np.float32(-1.0))
    return float(np.nextafter(s[k], np.float32(np.inf)))


def evaluate(emb, ids, sessions, threshold=None):
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    e = e.astype(np.float32)
    n = len(e)
    sim = e @ e.T
    iu = np.triu_indices(n, 1)
    s = sim[iu]
    same = (ids[:, None] == ids[None, :])[iu]
    gen, imp = s[same], s[~same]
    # EER on the exact score set
    thr = np.sort(np.unique(s))
    gs, ims = np.sort(gen), np.sort(imp)
    far = 1 - np.searchsorted(ims, thr, side="left") / max(len(ims), 1)
    frr = np.searchsorted(gs, thr, side="left") / max(len(gs), 1)
    k = int(np.argmin(np.abs(far - frr)))
    out = {
        "images": int(n), "palms": int(len(set(ids.tolist()))),
        "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp)),
        "eer": float((far[k] + frr[k]) / 2), "eer_threshold": float(thr[k]),
        "tar_at_far": {},
        "genuine": gen, "impostor": imp,
    }
    for a in (1e-2, 1e-3, 1e-4):
        t = far_threshold(imp, a)
        out["tar_at_far"][str(a)] = {"tar": float((gen >= t).mean()) if len(gen) else None, "threshold": t}
    # rank-1: second session as probes, first as gallery (else nearest other image)
    np.fill_diagonal(sim, -np.inf)
    us = sorted(set(sessions.tolist()))
    if len(us) >= 2:
        g, p = sessions == us[0], sessions == us[1]
        best = np.flatnonzero(g)[np.argmax(sim[np.ix_(p, g)], axis=1)]
        out["rank1"] = float((ids[best] == ids[p]).mean())
        out["rank1_kind"] = f"{us[1]} probes vs {us[0]} gallery"
    else:
        out["rank1"] = float((ids[np.argmax(sim, axis=1)] == ids).mean())
        out["rank1_kind"] = "nearest other image"
    if threshold is not None:
        tp, fp = int((gen >= threshold).sum()), int((imp >= threshold).sum())
        fn, tn = len(gen) - tp, len(imp) - fp
        out["at_threshold"] = {"threshold": threshold, "tar": tp / max(len(gen), 1), "far": fp / max(len(imp), 1),
                               "mcc": mcc(tp, fp, fn, tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out
