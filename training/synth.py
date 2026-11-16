"""Synthetic hand images for an end-to-end smoke test of the pipeline (NOT for accuracy claims).

Each identity gets its own set of palm lines; each image jitters position, rotation, line width,
brightness and noise. File layout and naming match the Tongji-style dataset the repo expects:
<out>/session1/00001.png … (10 images per palm per session).
"""
import argparse, os

import cv2
import numpy as np


def hand(rng, lines, jit):
    h, w = 480, 640
    img = np.full((h, w), 20, np.uint8)
    cx, cy = w // 2 + jit.integers(-25, 26), h // 2 + jit.integers(-20, 21)
    cv2.ellipse(img, (int(cx), int(cy)), (150, 175), 0, 0, 360, int(jit.integers(165, 200)), -1)
    for k in range(4):  # fingers
        x = int(cx - 105 + 70 * k)
        cv2.rectangle(img, (x - 22, int(cy - 330)), (x + 22, int(cy - 120)), int(jit.integers(160, 195)), -1)
    for (x0, y0, x1, y1, xm, ym) in lines:  # identity-specific creases (quadratic curves)
        t = np.linspace(0, 1, 40)[:, None]
        p0, pm, p1 = np.array([x0, y0]), np.array([xm, ym]), np.array([x1, y1])
        pts = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * pm + t ** 2 * p1) + [cx, cy]
        cv2.polylines(img, [pts.astype(np.int32)], False, int(jit.integers(70, 110)), int(jit.integers(2, 4)))
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(jit.uniform(-8, 8)), 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=20)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img.astype(np.float32) * jit.uniform(0.85, 1.1) + jit.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--palms", type=int, default=60)
    ap.add_argument("--per", type=int, default=10)
    a = ap.parse_args()
    for s in ("session1", "session2"):
        os.makedirs(os.path.join(a.out, s), exist_ok=True)
    for pid in range(a.palms):
        rng = np.random.default_rng(1000 + pid)
        lines = [tuple(rng.uniform(-120, 120, 6)) for _ in range(int(rng.integers(4, 7)))]
        for si, s in enumerate(("session1", "session2")):
            for k in range(a.per):
                jit = np.random.default_rng(10_000_000 + pid * 1000 + si * 100 + k)
                n = pid * a.per + k + 1
                cv2.imwrite(os.path.join(a.out, s, f"{n:05d}.png"), hand(rng, lines, jit))
    print(f"wrote {a.palms} palms × 2 sessions × {a.per} images to {a.out}")


if __name__ == "__main__":
    main()
