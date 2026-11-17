# palmprint (Rust inference)

A Rust port of this prototype's inference path, from image to match:
preprocessing → trained ResNet-18 encoder (ONNX) → cosine top-k against `output/embeddings/`.
It replaces `app.py` (Streamlit), `client/*.py` and `preprocess/*.py`. Training stays in Python:
`training/` trains and exports the encoder plus a model card (calibrated threshold, test split,
metrics) that every command here reads; see [`training/README.md`](../training/README.md).
`palmprint verify-model` checks that the ONNX file is exactly what the trainer exported.

**Palm ridges and faces** (from the `palmid` crate, [dhaneshkk/test1_palm](https://github.com/dhaneshkk/test1_palm)):

* `build-db` also writes palm-ridge codes (`ridge_codes.npy`): Competitive Code (orientation of the
  strongest dark-line Gabor response, 28×28 cells) on the same 138×138 ROI the CNN sees. Matching
  ranks the database (up to 20 000 images) by `w · cosine + (1 − w) · (1 − ridge distance)`. The
  weight and threshold come from Python (`pair-scores` → `training/calibrate_fusion.py` → model
  card); without a calibrated fusion the CNN ranks alone and ridge distances are only reported.
  `eval` reports CNN, ridge-only and fused metrics on the same test pairs.
* `serve` also finds faces in the uploaded image (YuNet, MIT, `models/NOTICE-yunet.md`), runs
  MediaPipe's 468-point face mesh and draws its Delaunay triangulation. Display only: faces are never
  matched. The mesh model is embedded in the `palmid` crate.

```bash
pip install onnxruntime                       # provides libonnxruntime (loaded at run time)
export ORT_DYLIB_PATH=$(python3 -c "import onnxruntime,os,glob;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
cd rust && cargo build --release
B=target/release/palmprint

$B serve                                      # web UI at http://127.0.0.1:8501 + /v1/chat/completions
$B match some_palm.tiff                       # top-5 + match verdict (threshold 0.80, as app.py)
$B embed some_palm.png                        # 512-d embedding as JSON
$B eval                                       # EER, TAR@FAR, MCC at the threshold, rank-1 (card's test palms when a card exists)
$B verify-model                               # SHA-256 + embedding parity vs the Python export (needs the card)
$B pair-scores --split cal --out ../data/cal_pairs.npy   # CNN + ridge scores per pair, for Python fusion calibration
$B build-db ../dataset/archive/session1 ../dataset/archive/session2 --roi-dir ../dataset/preprocessed_images
$B serve --roi-dir ../dataset/preprocessed_images   # match thumbnails, like the Streamlit app
```

`models/palmprint_encoder_512.onnx` is exported from `output/model/palmprint_encoder.pth` by
`tools/export_onnx.py` (max difference vs PyTorch 1.4e-6). It outputs the 512-d pooled features that
`app.py` actually uses.

## Parity with the Python

`cargo test` checks the port against outputs of the repo's own `preprocess/preprocessor.py`
(`tools/make_fixtures.py`) on three images: the README sample, a synthetic hand, and a hand at the
bottom-right edge, where the ROI is truncated to 276×243:

| Stage | Result |
|---|---|
| gray (`COLOR_RGB2GRAY` applied to BGR, a quirk kept on purpose), CLAHE 5×5, threshold 80 | bit-identical |
| `findContours(RETR_EXTERNAL)`: number of contours, largest area, moments, centroid | identical |
| ROI crop (window shifted at left/top edges, truncated at right/bottom, as in the Python) | bit-identical |
| resize to 138×138 (including OpenCV's 2×-shrink fast path) | bit-identical |
| embedding vs onnxruntime-Python | identical (max abs Δ = 0) |
| reading the committed `all_embeddings.npy` / `image_names.npy` | 11,996 × 512 |

JPEG decoding can differ by a grey level from OpenCV's libjpeg-turbo. TIFF and PNG are exact.

**Threshold:** `--threshold` if given, else the model card's calibrated threshold, else 0.80 (as
`app.py`). The card sits next to the model as `<model>.card.json`. The committed
`palmprint_encoder_512.onnx` (exported from the prototype's `.pth`) has no card, so it keeps 0.80.

`lean/Match.lean` proves eight properties (Lean 4.31, kernel only), and `src/lean_mirror.rs` checks
them on the same arithmetic. The training-side threshold rule: at most ⌊α·N⌋ calibration impostor
scores reach the threshold the trainer picks, so the calibration FAR is ≤ α (`far_threshold_bound`,
`far_le_alpha`). Fusion: a higher cosine or a lower ridge distance never lowers the fused score,
and the weight's ends are the CNN alone and the ridges alone (`fuse_mono`, `fuse_ends`). The match rule:
* the result list is never empty;
* the below-threshold fallback never turns "no match" into "match";
* a higher threshold never adds a match;
* when a match is found, every returned score is at or above the threshold.

## What the evaluation shows (`palmprint eval`)

The image names carry identity: 10 consecutive images per palm per session (600 palms × 2 sessions).
Images in the same 10-file block average cosine 0.752; images in the next block average 0.487. On the
committed embeddings, over all 71.8 M pairs:

| Metric | Value |
|---|---|
| EER | 14.1 % (at cosine 0.589) |
| TAR @ FAR 1 % / 0.1 % / 0.01 % | 55.8 % / 35.5 % / 22.5 % |
| at the app's 0.80 threshold | TAR 41.3 %, FAR 0.216 %, MCC 0.31 |
| rank-1, session-2 probe vs session-1 gallery | 33.4 % |
| rank-1, nearest other image in the database | 92.6 % (92 % of those nearest neighbours are from the same session) |

At FAR 0.216 %, a query against the 11,996-image database clears the 0.80 threshold for about 26
impostor images on average. A synthetic drawing of a hand (`tests/fixtures/synthetic_input.png`) gets
"match found" at 0.84.


## Web app (`web/`, training + inference)

`palmprint-web` is a [Topcoat](https://github.com/tokio-rs/topcoat) 0.12 app (Rust ≥ 1.99) that runs
the same steps as the CLI and the Python scripts, from the browser, with live status:

```
 Train form ─► 1 prepare.py (Python) ─► 2 train.py, SAM (Python; epoch loss + calibration EER live)
           ─► 3 verify (Rust) ─► 4 build database (Rust, progress) ─► 5 pair scores (Rust) → calibrate_fusion.py (Python)
           ─► 6 evaluate on test palms (Rust) ─► swap the live engine  ── only Evaluating → Ready (lean/Pipeline.lean)
 Upload ──► live engine ─► matches (score, CNN cosine, ridge distance), palm region, face mesh
```

```bash
cargo install topcoat-cli --version 0.12.1 --locked
topcoat asset bundle -p palmprint-web --release
export ORT_DYLIB_PATH=…   # as above
PP_PYTHON=$(which python3) cargo run -p palmprint-web --release        # http://127.0.0.1:3000
```

`cargo build` in `rust/` still builds only the CLI (`default-members`), so it keeps working on older
Rust; the web app is `-p palmprint-web`.

| env | default | |
|---|---|---|
| `PP_REPO` | `..` | repository root (where `training/` is) |
| `PP_WORK` | `palmprint_work` | trained model, card, trained database, intermediate files, demo data |
| `PP_PYTHON` | `python3` | interpreter with `training/requirements.txt` installed |
| `PP_DATA` | — | image folders pre-filled in the form (comma-separated) |
| `PP_MODEL` | `models/palmprint_encoder_512.onnx` | model served until something is trained |
| `PP_ROI_DIR` | — | preprocessed ROIs (`<name>.npy`) for match thumbnails |
| `PP_THREADS` | `4` | onnxruntime and torch threads |
| `HOST` / `PORT` | `127.0.0.1` / `3000` | Topcoat |

* The committed model and database serve until a run finishes; a finished run (model + card with
  fusion + database in `PP_WORK`) is loaded again on the next start.
* **Make demo data** writes synthetic hands (`training/synth.py`) so the pipeline can be tried without
  the dataset; leave the folders field empty to train on them.
* **Cancel** stops the Python child and marks the run failed; the live engine is untouched.
* `POST /api/match` (used by the page), `POST /v1/chat/completions` (OpenAI-compatible; no image →
  live status and the last results) and `GET /v1/models` run on the same live engine.
* With fusion, the whole database (up to 20 000 images) is ranked by the fused score, so a low CNN
  weight isn't undercut by a cosine pre-filter.

Checked end to end on synthetic hands (60 palms, 6 epochs from scratch, 2 CPU threads, ≈ 9 min):
the pipeline ran through all six phases, the weak 6-epoch CNN (test EER 36 %) led calibration to
choose ridges only (w = 0), test MCC 0.24 for the CNN vs 0.68 fused, and an uploaded image matched
its own palm's other images.
