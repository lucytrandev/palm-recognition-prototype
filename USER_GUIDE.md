# Palmprint recognition user guide

How to train the palmprint model, build the palm database, check how well it works, and use the web
page and API. Training runs in Python; everything after training runs in the Rust program `palmprint`.
For background, see the [README](README.md).

* [1. What you get](#1-what-you-get)
* [2. The fastest path: one Colab cell](#2-the-fastest-path-one-colab-cell)
* [2b. Everything in the browser: the web app](#2b-everything-in-the-browser-the-web-app)
* [3. Running it on your own machine](#3-running-it-on-your-own-machine)
* [4. Train the model](#4-train-the-model)
* [5. Check the model and build the database](#5-check-the-model-and-build-the-database)
* [6. Combine with palm lines (fusion)](#6-combine-with-palm-lines-fusion)
* [7. Read the evaluation](#7-read-the-evaluation)
* [8. The web page](#8-the-web-page)
* [9. The API](#9-the-api)
* [10. Everyday tasks](#10-everyday-tasks)
* [11. Troubleshooting](#11-troubleshooting)
* [12. Safe use](#12-safe-use)

---

## 1. What you get

* **Match a palm image** against a database of palm images: the 5 most similar, each with a score,
  and a yes/no "match found".
* Scores combine a **neural-network embedding** with **palm-line (ridge) codes**, weighted by numbers
  chosen on palms the model never trained on.
* If the uploaded image contains a **face**, it is shown with a 468-point triangle mesh. Faces are
  never matched.
* An **evaluation report** on unseen palms: error rates and MCC (1 = perfect, 0 = guessing).

## 2. The fastest path: one Colab cell

1. Open [`training/train_colab.ipynb`](training/train_colab.ipynb) in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.**
3. Data: either
   * add your Kaggle credentials as Colab secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` (the cell downloads
     the [palm dataset](https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset)), or
   * set `DATA_DIRS` at the top of the cell to your own `session1` / `session2` folders (e.g. on Drive), or
   * set `SYNTHETIC_SMOKE = True` to try the pipeline on drawn hands (results mean nothing about accuracy).
4. Run the cell. It prints, in order: training progress per epoch, Python's test results, the Rust
   build, the model check, the database build, the chosen fusion, Rust's test results for CNN, ridges
   and fused, an API answer, and `LEAN VERIFIED`.

To keep the results, download `rust/models/palmprint_encoder_512.onnx`, the `.card.json` and the
`.parity_*.npy` files next to it.

## 2b. Everything in the browser: the web app

The web app runs training and matching from one page, with live progress. Set up §3 first (Python
environment and `ORT_DYLIB_PATH`), then, with **Rust 1.99 or newer** (`rustup update`):

```bash
cd rust
cargo install topcoat-cli --version 0.12.1 --locked
topcoat asset bundle -p palmprint-web --release
PP_PYTHON=$(which python3) cargo run -p palmprint-web --release
```

Open **http://127.0.0.1:3000**.

**Train (right side).**

1. **folders**: your session folders, comma-separated (e.g. `/data/palms/session1,/data/palms/session2`).
   No dataset at hand? Press **Make demo data** and leave the field empty: it trains on drawn hands,
   which only shows the pipeline works.
2. **epochs** (40 is a start on a GPU; on a laptop CPU try 6–10 first), **loss** `supcon` (uses the
   palm identities) or `simclr` (the original self-supervised loss), **target FAR** (0.001 = 0.1 % false
   accepts), **ImageNet start** (needs internet once).
3. **Start training.** The Pipeline panel walks through six phases: preparing → training (loss and
   calibration error per epoch are charted) → verifying → building the database (progress bar) →
   fusion → evaluating. **Cancel** stops it; the live model isn't touched.
4. When it ends, the **Unseen test palms** table shows CNN, palm ridges and combined: error rate, true
   accepts at 0.1 % false accepts, true/false accepts at the threshold, MCC and rank-1 (§7 explains
   each). Only then does the **Live model** panel switch to the newly trained model.

**Match (left side).** Choose a palm image: the result shows *match found* or *no match*, the palm
region the model saw, any faces with their mesh, and the top 5 matches with the combined score, the
network's cosine and the palm-line distance.

Trained files go to `rust/palmprint_work/` (change with `PP_WORK`); the app reloads the last
finished run on its next start. Other settings are listed in [rust/README.md](rust/README.md#web-app-web-training--inference).

## 3. Running it on your own machine

You need Python 3.10+, Rust (stable), git and, for training, ideally an NVIDIA GPU.

```bash
git clone -b rust-inference https://github.com/dhaneshkk/palm-recognition-prototype
cd palm-recognition-prototype
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh     # if Rust isn't installed
```

In every new terminal, from the repository folder:

```bash
source .venv/bin/activate
export ORT_DYLIB_PATH=$(python3 -c "import onnxruntime,os,glob;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
```

**Dataset layout.** One folder per capture session, images named by number, 10 consecutive images per
palm (Tongji-style): `dataset/archive/session1/00001.tiff … 06000.tiff`, the same for `session2`.
Images 1–10 are palm 1, 11–20 palm 2, and so on, in both sessions. Use `--block` if your dataset has a
different number of images per palm.

## 4. Train the model

```bash
python training/prepare.py dataset/archive/session1 dataset/archive/session2 -o data/rois.npz
python training/train.py data/rois.npz --out rust/models --epochs 40
```

* `prepare.py` cuts the palm region from every image exactly as the Rust program will, and reports how
  many images failed (no palm region found).
* `train.py` prints one line per epoch, with the loss and the error rate on the calibration palms. It
  keeps the best epoch, picks the match threshold, scores the test palms and exports the model.

Useful options: `--epochs`, `--target-far` (default 0.001 = 0.1 % false accepts), `--loss simclr`
(the original self-supervised loss instead of using palm labels), `--no-pretrained` (offline).

Training writes `rust/models/palmprint_encoder_512.onnx`, replacing the original model, plus
`palmprint_encoder_512.card.json` (threshold, which palms are test palms, metrics) and two parity files.
**After training, always rebuild the database** (next section): old embeddings don't fit a new model.

## 5. Check the model and build the database

```bash
cd rust
cargo build --release
B=target/release/palmprint

$B verify-model
$B build-db ../dataset/archive/session1 ../dataset/archive/session2
```

* `verify-model` confirms the Rust program loads exactly the file Python exported and computes the same
  embeddings. It ends with `"ok": true`; anything else means the files don't belong together.
* `build-db` embeds every image and stores palm-line codes. By default it **overwrites**
  `output/embeddings/` (the committed database); use `--embeddings`, `--names` and `--ridges` to write
  elsewhere, and pass the same three options to every later command.

## 6. Combine with palm lines (fusion)

```bash
$B pair-scores --split cal --out ../data/cal_pairs.npy
python ../training/calibrate_fusion.py ../data/cal_pairs.npy models/palmprint_encoder_512.card.json
```

Rust scores every pair of calibration images both ways (network and palm lines); Python tries weights
from 0 (palm lines only) to 1 (network only), keeps the one with the lowest error, sets the combined
threshold and writes both into the model card. From then on, matching uses the combined score. Skip
this step and matching uses the network alone.

## 7. Read the evaluation

```bash
$B eval > ../data/eval.json
```

With a model card, `eval` uses only the **test palms**, which training never saw. The report has
up to four parts: `report` (network, the original method), `cnn_exact`, `ridge` and `fused`. In each:

| Field | Meaning | Better |
|---|---|---|
| `eer` | equal error rate: where false accepts and false rejects are equal | lower |
| `tar_at_far` | share of genuine pairs accepted when false accepts are held at 1 %, 0.1 %, 0.01 % | higher |
| `at_threshold.tar` / `.far` | genuine accepted / impostors accepted at the threshold actually used | high / low |
| `at_threshold.mcc` | one-number quality of the accept/reject decision, −1 to 1 | higher |
| `rank1_cross_session` | a session-2 image's best match in session 1 is the same palm | higher |

Compare `fused` with `cnn_exact`: if fused isn't better, the fusion weight will usually come out close
to 1 anyway.

## 8. The web page

```bash
$B serve                      # open http://127.0.0.1:8501
$B serve --roi-dir ../dataset/preprocessed_images   # optional: thumbnails of the matched palms
```

Upload a palm image (JPG, PNG, TIFF). The page shows:

* the processing time, where the palm region was found and the scoring used (`cnn` or
  `cnn+ridges (w=…)`);
* the cropped palm region the model saw;
* faces, if any, with the triangle mesh;
* the top 5 matches: combined score, and with fusion also the network score and palm-line distance;
* **match found** or not, against the threshold.

If no palm region is found but there is a face, you get the face mesh and a "faces only" note.

## 9. The API

`serve` also answers OpenAI-style requests:

```bash
curl -s localhost:8501/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "palmprint",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 palm.png)"'"}}]}]}'
```

`choices[0].message.content` is JSON text with `results`, one per image: `match_found`, `threshold`,
`scoring`, `matches` (name, score, cosine, ridge distance), `faces` (box, presence, number of points
and triangles) and the preprocessing trace.

## 10. Everyday tasks

| Task | How |
|---|---|
| Match one image from the terminal | `$B match path/to/palm.png` |
| Add new palm images | put them in a session folder and rerun `build-db` on all folders (it rebuilds from scratch) |
| Use a stricter or looser threshold for one run | `--threshold 0.9` on `match`, `eval` or `serve` |
| Evaluate on the calibration palms or everything | `$B eval --split cal` or `--split all` |
| Go back to the original model | `git checkout rust/models/palmprint_encoder_512.onnx output/embeddings` and delete the `.card.json` |
| Check the proofs | install Lean 4.31 and run `lean rust/lean/Match.lean` (the notebook does this) |

## 11. Troubleshooting

| Problem | What to do |
|---|---|
| Errors mentioning `libonnxruntime` | run the `export ORT_DYLIB_PATH=...` line (§3) in this terminal |
| Colab can't download the dataset | add `KAGGLE_USERNAME` / `KAGGLE_KEY` secrets, or set `DATA_DIRS` |
| `verify-model` not ok | the `.onnx`, `.card.json` and `.parity_*` files come from different runs; copy all of them from one run |
| "ridge codes and embeddings differ in count" | the database is half old, half new: rerun `build-db` |
| "--split needs a model card" | there's no `.card.json` next to the model (original model): use `--split all` |
| "could not preprocess the image" | no bright palm region on a dark background was found; the method expects images like the dataset's |
| Many failures in `prepare.py` | the images don't have a dark background around the palm; the preprocessing is tuned for this dataset |
| Fused results no better | expected on some data; the card's `fusion.calibration.eer_by_weight` shows the trade-off |
| Training very slow | use a GPU (Colab T4); on CPU, reduce `--epochs` |
| Web app: "asset bundle missing" | run `topcoat asset bundle -p palmprint-web --release` in `rust/` |
| Web app build says Rust is too old | `rustup update` (Topcoat needs 1.99 or newer); the CLI alone builds on older Rust |
| Web app training fails at once with "No module named …" | start it with `PP_PYTHON` pointing at the Python that has `training/requirements.txt` installed |
| Combined weight comes out 0 (palm lines only) | the network is under-trained; train longer or with ImageNet start |

## 12. Safe use

* Only add palm images of people who agreed, and remove them when asked (delete the images, rerun
  `build-db`).
* There's no liveness check, and accuracy of the new pipeline on the real dataset hasn't been
  published yet: don't use it to grant access or approve payments.
* `serve` listens only on this computer by default and has no login. Don't expose it to a network
  without adding authentication.
