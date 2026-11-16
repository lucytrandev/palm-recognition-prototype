# Training (Python) → inference (Rust)

Training runs in Python. Everything at run time (preprocessing, encoder, matching, web UI, API) runs in
the Rust crate in `../rust`. The two halves meet in three files that the trainer writes into
`rust/models/`:

| File | What it is | Checked by |
|---|---|---|
| `palmprint_encoder_512.onnx` | ResNet-18 backbone: `input` [b,3,138,138] → `embedding` [b,512] | `palmprint verify-model` (SHA-256) |
| `palmprint_encoder_512.card.json` | calibrated threshold, target FAR, identity split, metrics, training config | read by every Rust command |
| `.parity_input.npy` / `.parity_embedding.npy` | 8 test ROIs and onnxruntime-Python embeddings | `palmprint verify-model` (max abs diff) |

Training sees exactly what inference sees: `prepare.py` uses `preprocess/preprocessor.py`, which the
Rust port reproduces bit-exactly (`rust/tests/parity.rs`).

## Run

```bash
pip install -r training/requirements.txt
python training/prepare.py dataset/archive/session1 dataset/archive/session2 -o data/rois.npz
python training/train.py data/rois.npz --out rust/models --epochs 40      # GPU recommended

cd rust && cargo build --release
B=target/release/palmprint
$B verify-model                                          # Rust loads exactly what Python exported
$B --embeddings ../data/emb.npy --names ../data/names.npy build-db ../dataset/archive/session1 ../dataset/archive/session2
$B --embeddings ../data/emb.npy --names ../data/names.npy eval     # test palms, card threshold
$B --embeddings ../data/emb.npy --names ../data/names.npy serve    # UI + /v1/chat/completions
```

Or run everything, including the Rust build, the API call and the Lean check, in the single-cell Colab
notebook `training/train_colab.ipynb`.

## What `train.py` does

1. **Open-set split by palm** (seeded): 70 % of palms train, 15 % calibrate, 15 % test. Test palms are
   never seen in training, which is how the matcher is used (new people enroll).
2. **Loss:** supervised contrastive (SupCon) over P palms × K images per batch, two augmented views
   each, with the prototype's augmentations. The dataset's names give identity (10 consecutive images
   per palm per session), so labels are free. `--loss simclr` keeps the prototype's self-supervised
   NT-Xent.
3. **Optimiser:** AdamW wrapped in **SAM** (sharpness-aware minimisation, ρ = 0.05; BatchNorm statistics
   frozen during the perturbed pass), cosine learning-rate decay.
4. **Model selection:** the epoch with the lowest EER on the calibration palms.
5. **Threshold:** the smallest float32 above the ⌊α·N⌋-th largest calibration impostor score
   (α = `--target-far`, default 0.1 %). At most ⌊α·N⌋ calibration impostor pairs then reach it:
   proved in `rust/lean/Match.lean` (`far_threshold_bound`, `far_le_alpha`) and checked on the same
   arithmetic in Rust (`lean_mirror::far_threshold`).
6. **Test metrics** on the unseen palms: EER, TAR @ FAR 1 / 0.1 / 0.01 %, accept/reject **MCC** over all
   pairs at the calibrated threshold, cross-session rank-1. Rust's `eval` recomputes them from its own
   embeddings for the card's test palms.

The exported embedding is the 512-d backbone output, as before; the projection head is used only for
the loss. Inputs are gray/255 replicated to 3 channels with no mean/std, as in the prototype.

## Smoke test (synthetic hands, not an accuracy claim)

`synth.py` draws hands whose palm lines depend on identity, with jittered pose, brightness and noise.
60 palms × 2 sessions × 10 images, CPU, from scratch, 15 epochs (≈ 9 min on 2 vCPU):

| | Python (card) | Rust (`eval --split test`) |
|---|---|---|
| test palms / images | 9 / 180 (1,710 genuine, 14,400 impostor pairs) | same |
| EER | 2.10 % | 2.10 % |
| TAR @ FAR 0.1 % (test-set threshold) | 93.0 % | 93.0 % |
| calibrated threshold (from 9 calibration palms) | 0.99096 | read from the card |
| at that threshold: TAR / FAR / MCC | 33.6 % / 0 % / 0.558 | 33.6 % / 0 % / 0.558 (identical counts) |
| cross-session rank-1 | 100 % | 100 % |

`verify-model`: SHA-256 matches the card, max |Δ| = 0 over the parity inputs. The Rust and Python
numbers are computed independently (Rust embeds the raw images itself) and agree exactly.

The calibrated threshold is much stricter than the test-set 0.1 % FAR point (0.976) because 9
calibration palms give few high impostor scores to calibrate on; with the real dataset's ~90
calibration palms it will sit much closer. A 4-epoch run gave scores bunched near 1 (threshold
0.99995) and one pair differed between Rust and Python by float rounding at that threshold.

Real accuracy needs the real dataset (Kaggle `saqibshoaibdz/palm-dataset`); it was not reachable from
the build environment.
