"""Reference outputs for the Rust parity tests, produced with the repo's own preprocess/preprocessor.py.
Usage (from the repo root): python rust/tools/make_fixtures.py rust/models/palmprint_encoder_512.onnx rust/tests/fixtures"""
import sys, os, json, numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, os.getcwd())
import preprocess.preprocessor as P

onnx_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

def synthetic(seed, w=800, h=600, cx=400, cy=300):
    """Dark background + bright hand-like blob with crease lines + small bright specks (extra contours)."""
    rng = np.random.default_rng(seed)
    img = (rng.normal(20, 6, (h, w, 3))).clip(0, 255).astype(np.uint8)
    hand = np.zeros((h, w), np.uint8)
    cv2.ellipse(hand, (cx, cy + 40), (170, 210), 0, 0, 360, 255, -1)
    for k in range(4):
        cv2.rectangle(hand, (cx - 150 + k * 85, cy - 330), (cx - 95 + k * 85, cy - 80), 255, -1)
    cv2.ellipse(hand, (cx - 210, cy + 40), (40, 120), 35, 0, 360, 255, -1)
    tex = (rng.normal(150, 12, (h, w))).clip(0, 255)
    for _ in range(9):
        p = rng.integers([cx - 150, cy - 60], [cx + 150, cy + 200], (2, 2))
        cv2.line(hand, tuple(map(int, p[0])), tuple(map(int, p[1])), 120, 3)
    g = np.where(hand > 0, np.where(hand == 255, tex, tex * 0.75), 0).astype(np.uint8)
    img = np.maximum(img, np.dstack([g, (g * 0.8).astype(np.uint8), (g * 0.7).astype(np.uint8)]))
    for _ in range(6):
        cv2.circle(img, tuple(int(v) for v in rng.integers([0, 0], [w, h])), int(rng.integers(2, 8)), (200, 200, 200), -1)
    return img

cases = {
    'original': cv2.imread('output/original_image.png'),
    'synthetic': synthetic(1),
    'edge': synthetic(2, cx=760, cy=600),          # centroid near the border → clipped, non-square ROI
}
index = {}
for name, bgr in cases.items():
    cv2.imwrite(f'{out}/{name}_input.png', bgr)
    bgr = cv2.imread(f'{out}/{name}_input.png')     # exactly what preprocess_image would read
    gray = P.to_grayscale(bgr)
    enh = P.reduce_noise(P.enhance_contrast_clahe(gray))
    binary, contours = P.segment_palm(enh)
    roi = P.extract_roi_centroid(enh, contours)
    largest = max(contours, key=cv2.contourArea); M = cv2.moments(largest)
    resized = P.resize_roi(roi)
    norm = P.normalize_roi(resized)
    full = P.preprocess_image(f'{out}/{name}_input.png')
    assert np.array_equal(full, norm)
    x = np.repeat(norm[None, None], 3, 1).astype(np.float32)
    emb = sess.run(None, {'input': x})[0][0]
    for k, v in [('gray', gray), ('enhanced', enh), ('binary', binary), ('roi', roi), ('resized', resized)]:
        cv2.imwrite(f'{out}/{name}_{k}.png', v)
    np.save(f'{out}/{name}_embedding.npy', emb.astype(np.float32))
    index[name] = dict(n_contours=len(contours), largest_area=float(cv2.contourArea(largest)), m00=M['m00'], m10=M['m10'], m01=M['m01'],
                       cx=int(M['m10'] / M['m00']), cy=int(M['m01'] / M['m00']), roi_shape=list(roi.shape))
    print(name, index[name])
json.dump(index, open(f'{out}/index.json', 'w'), indent=1)
