# Palmprint Recognition: Python training, Rust inference

Palmprint matching for authentication research. Upload a palm image and get the most similar palms in
the database, scored by a CNN embedding fused with palm-ridge (palm-line) codes. If the image contains a
face, it is shown with a 468-point triangulated mesh (display only, never matched).

This is a fork of [lucytrandev/palm-recognition-prototype](https://github.com/lucytrandev/palm-recognition-prototype)
by Lucy Tran. Her original Python/Streamlit prototype is described [further down](#original-prototype-lucy-tran-2025);
this fork (branch `rust-inference`) splits it into a Python training pipeline and a Rust inference program.

**New here? Read the [user guide](USER_GUIDE.md)**: training in Colab, building the database,
calibrating, evaluating, the web page and the API, step by step.

## How it fits together

```
 Python (training/)                                     Rust (rust/, inference only)
 ──────────────────                                     ────────────────────────────
 palm images ─► prepare.py (same preprocessing as Rust)
             ─► train.py: SupCon + SAM, open-set split ─► model.onnx + model card ─► verify-model
                  threshold at target FAR, test MCC                                   build-db: embeddings
                                                                                         + ridge codes
 calibrate_fusion.py ◄──── calibration-pair scores ◄───────────────────────────────── pair-scores
   fusion weight + threshold ───► model card ───────────────────────────────────────► eval (CNN / ridges / fused)
                                                                                       serve: web page + /v1/chat/completions
```

* **Training (Python):** identity-disjoint split (70 % train / 15 % calibration / 15 % test palms),
  supervised-contrastive loss (or the original NT-Xent), AdamW with **SAM**, model selection by
  calibration EER, threshold at a target false-accept rate, EER / TAR@FAR / **MCC** / cross-session
  rank-1 on unseen palms. Exports ONNX, a model card and parity fixtures. See [training/README.md](training/README.md).
* **Inference (Rust):** bit-exact port of the preprocessing, ONNX encoder, cosine matching re-ranked
  with palm-ridge codes (CompCode) when a calibrated fusion is in the card, evaluation, a web page and
  an OpenAI-compatible API. See [rust/README.md](rust/README.md).
* **One notebook:** [`training/train_colab.ipynb`](training/train_colab.ipynb) runs everything in a
  single Colab cell: data, training, Rust build, model check, database, fusion, test evaluation, an API
  call and the Lean check.
* **Proofs:** `rust/lean/Match.lean` (Lean 4.31, no `sorry`): match-rule properties, the threshold
  rule keeps the calibration FAR ≤ target (`far_threshold_bound`, `far_le_alpha`), and the fused score
  never drops when the CNN or ridge match improves (`fuse_mono`, `fuse_ends`).

## Quick start

```bash
# training (GPU recommended) — or just run training/train_colab.ipynb in Colab
pip install -r training/requirements.txt
python training/prepare.py dataset/archive/session1 dataset/archive/session2 -o data/rois.npz
python training/train.py data/rois.npz --out rust/models --epochs 40

# inference
cd rust && cargo build --release
export ORT_DYLIB_PATH=$(python3 -c "import onnxruntime,os,glob;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
B=target/release/palmprint
$B verify-model
$B build-db ../dataset/archive/session1 ../dataset/archive/session2
$B pair-scores --split cal --out ../data/cal_pairs.npy
python ../training/calibrate_fusion.py ../data/cal_pairs.npy models/palmprint_encoder_512.card.json
$B eval                      # CNN, ridges and fused on the unseen test palms
$B serve                     # http://127.0.0.1:8501 and /v1/chat/completions
```

Without training, `cd rust && cargo build --release && target/release/palmprint serve` runs the
original model against the committed database (CNN only, threshold 0.80, as the Streamlit app).

## Status and limits

* **Accuracy of the new pipeline on the real dataset is not measured yet**: Kaggle wasn't reachable
  from the build environment. Everything was run end to end on synthetic hands (`training/synth.py`);
  those numbers check the pipeline, not biometric accuracy. The original model's numbers on the
  committed embeddings are in [rust/README.md](rust/README.md) (EER 14.1 %).
* No liveness check; a printed palm could match. Not for access control as is.
* The repository has no licence file of its own (inherited from the upstream fork); check before reuse.
* Palm-ridge codes, the face detector and the face mesh come from
  [dhaneshkk/test1_palm](https://github.com/dhaneshkk/test1_palm) (`palmid` crate, pinned revision).

---

# Original prototype (Lucy Tran, 2025)

The sections below are the original author's description of the Python/Streamlit prototype that this
fork builds on (ResNet-18 + 256-d projection head, NT-Xent, Streamlit UI at threshold 80 %). They still
describe `app.py`, `self_supervised/` and `preprocess/`, which remain in the repository.

## Palmprint Recognition for Authentication using Self-Supervised Learning

### Project Goal and Approach


This project implements a palmprint recognition system for authentication purposes. Due to the lack of labeled data for training, a self-supervised learning approach was adopted to learn meaningful feature representations from a large dataset of unlabeled palmprint images. The system utilizes contrastive learning to train a deep neural network to extract discriminative features, enabling palmprint matching and similarity comparison.  Finally, a user-friendly web interface was built to demonstrate the system's functionality.

### Dataset Description
For this project, a dataset of **12,000 unlabeled palmprint images** was used. The images are in **TIFF format** and were **obtained from the "Kaggle Palm Recognition Dataset for Authentication System"** dataset available on Kaggle ([https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data](https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data)).

This dataset is described as containing high-quality images designed for training and evaluating palm recognition models for biometric authentication.  While the original dataset includes categories for palm detection, palm vein patterns, and palm print matching, this project utilizes the **palmprint images suitable for palm print matching**. The dataset consists of images with **relatively high quality, designed for real-world biometric authentication applications.** The images are expected to exhibit variations inherent in biometric data collection, but are generally intended to be high-quality for effective model training. Each image in the dataset is also associated with metadata within the original Kaggle dataset, though this project focuses on the image data itself for self-supervised learning.

*   **Number of images:** 12,000
*   **Type of images:** Palmprint images
*   **Labeled or Unlabeled:** Unlabeled data
*   **Image format:** TIFF images
*   **Source:**  [Palm Recognition Dataset for Authentication System from Kaggle](https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data)
*   **Characteristics:** Images typically feature a dark background, which aids in segmenting the palm region from the surroundings. The dataset likely contains multiple images of palmprints from the same individual, reflecting a realistic biometric authentication scenario and hinting at potential for future identity-based analysis.While designed to be high quality, the dataset likely includes natural variations in palm appearance due to individual differences, minor pose variations, and subtle changes in skin texture.

### Image Preprocessing

Before feature extraction, palmprint images undergo a preprocessing pipeline to enhance relevant features and standardize the input for the model. The key stages are:

1.  **Grayscale Conversion:** The input color palmprint image (if any) is converted to grayscale to simplify processing and focus on texture and line patterns.
2.  **Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization):**  CLAHE is applied to improve the local contrast of the palmprint image, making the lines and ridges more prominent and easier to detect. This uses a clip limit of `2.0` and a tile grid size of `(5, 5)`.
3.  **Noise Reduction (Median Blur):** Median Blur with a kernel size of `1` is used to reduce noise while preserving edges.  While a kernel size of `1` effectively means minimal blurring, this step is included for potential future noise reduction adjustments.
4.  **Palm Segmentation (Thresholding and Contour Extraction):**  A thresholding technique (using a `threshold_value` of `80`) is applied to create a binary mask, separating the palm region (foreground) from the background.  Contours are then extracted to identify the palm's outline.
5.  **Region of Interest (ROI) Extraction (Centroid-Based):**  The Region of Interest (ROI) is extracted by finding the centroid (center of mass) of the largest contour (assumed to be the palm). A fixed-size square ROI of `(276x276)` pixels, centered at this centroid, is then cropped from the contrast-enhanced and noise-reduced grayscale image.
6.  **ROI Resizing:** The extracted ROI is resized to a standardized `(138x138)` pixel size to ensure consistent input dimensions for the feature extraction model.
7.  **ROI Normalization:** Finally, the pixel values of the resized ROI are normalized to the range of `[0, 1]` by dividing by `255.0`. This step helps in stabilizing and accelerating the training of the deep learning model.

#### Example Preprocessing Stages

Below are example images illustrating the key stages of the preprocessing pipeline applied to a sample palmprint image from the dataset, displayed in a horizontal flow:

<table>
  <tr>
    <td align="center">
      (1) Original Image:<br>
      <img src="output/original_image.png" width="150"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (2) Grayscale Conversion:<br>
      <img src="output/Grayscale%20Image_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (3) Contrast Enhanced (CLAHE):<br>
      <img src="output/CLAHE%20Image_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (4) Noise Reduction: <br>
      <img src="output/Blur%20Image_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (5) Contour Extraction: <br>
      <img src="output/Contours_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (6) (ROI) Extraction: <br>
      <img src="output/ROI_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (7) ROI Resizing: <br>
      <img src="output/Resized%20ROI_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (8) ROI Normalization: <br>
      <img src="output/Normalized%20ROI_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (9) Final Preprocess Result: <br>
      <img src="output/Final%20Preprocess%20Image_screenshot_27.01.2025.png"><br>
    </td>
  </tr>
</table>


---
### Self-Supervised Learning Method Used

This project employed a **contrastive learning** approach, specifically inspired by **SimCLR (Simple Contrastive Learning for Representations)**.  Since the dataset is unlabeled, self-supervised learning was crucial to learn meaningful feature representations.

The SimCLR-inspired method works by:
1.  **Creating positive pairs:** For each palmprint image in a batch, two augmented versions are generated using various image transformations (see "Data Augmentation" section). These two augmented views of the same palmprint form a "positive pair."
2.  **Creating negative pairs:** Augmented views from different original palmprint images within the same batch are treated as "negative pairs."
3.  **Training objective:** The model is trained to learn feature representations (embeddings) such that:
    *   Embeddings of positive pairs are pulled closer together in the feature space.
    *   Embeddings of negative pairs are pushed further apart.

This contrastive objective forces the model to learn features that are invariant to the applied augmentations and discriminative between different palmprints, all without requiring explicit labels.

**Loss Function:** The training process utilizes the **NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss** function to achieve this objective. This loss function, with a **temperature parameter of 0.07**, encourages similar embeddings for positive pairs and dissimilar embeddings for negative pairs in a normalized embedding space. By minimizing this loss, the model learns to extract robust and discriminative features from the unlabeled palmprint images.

### Model Architecture

<table>
  <tr>
    <td>
      <p align="center">
        <a href="output/model/7.%20palmprint_encoder.onnx.svg">
          <img src="output/model/7.%20palmprint_encoder.onnx.svg" height="500">
        </a>
</p>
    </td>
    <td>
      <p>
The feature extraction model is based on a <strong>ResNet-18</strong> Convolutional Neural Network (CNN) architecture.  The <strong>encoder</strong> part of the model utilizes a <strong>ResNet-18 backbone</strong>, initialized with weights <strong>pre-trained on ImageNet</strong>.  The original classification head of the pre-trained ResNet-18 was removed, and a <strong>2-layer Multilayer Perceptron (MLP) projection head</strong> was added on top of the encoder's output features.
        <br><br>
Specifically, the projection head consists of two linear layers with a ReLU activation function in between. This projection head further processes the features extracted by the ResNet-18 encoder before outputting the final <strong>256-dimensional feature embeddings</strong>.
        <br><br>
This architecture was chosen to leverage the powerful feature extraction capabilities of the ResNet-18 architecture, which is pre-trained on a large image dataset (ImageNet). The projection head is used to learn effective representations suitable for the contrastive learning task, mapping the ResNet-18 features into a lower-dimensional embedding space optimized for similarity comparisons. The model is trained to output <strong>256-dimensional feature embeddings</strong> for each input palmprint image.
      </p>
    </td>
  </tr>
</table>


### Training Process

The self-supervised model was trained using PyTorch. The training process involved:

*   **Framework:** PyTorch
*   **Optimizer:** Adam optimizer with a learning rate of `1e-4` (0.0001)
*   **Loss Function:** NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss with a temperature of `0.07`
*   **Training Epochs:** 100 epochs
*   **Batch Size:** 64
*   **Data Augmentations:** During training, the following data augmentations were applied to generate positive pairs:
    *   `RandomResizedCrop`
    *   `RandomRotation` (small angles)
    *   `ColorJitter` (grayscale - brightness and contrast adjustments)
    *   `GaussianBlur`
*   **Training Procedure:** The training loss was monitored, and the model weights were updated using backpropagation with the Adam optimizer to minimize the NT-Xent loss. The model learned to create embeddings where augmented views of the same palmprint are close together, and views of different palmprints are far apart in the embedding space.

### UI Functionality

*   **Framework used:** Streamlit
*   **Key features:** Image upload, display of uploaded image, display of top matching palmprints from database, similarity scores, indication of match/no match.

A simple and interactive web user interface was developed using **Streamlit** to demonstrate the palmprint recognition system. The UI provides the following functionalities:

*   **Image Upload:** Users can upload a palmprint image file (JPG, JPEG, PNG, TIFF).
*   **Uploaded Image Display:** The uploaded palmprint image is displayed on the UI.
*   **Matching Results Display:**  Upon uploading an image, the system compares it against a database of 11996 images pre-extracted palmprint features.
*   **Top Matching Palmprints:** The UI displays the top matches of 5 most similar palmprint images from the database, along with their cosine similarity scores.
*   **Match Indication:** The UI indicates whether a "match" is found based on a predefined similarity threshold value of 80%.

The UI allows users to easily test the palmprint recognition system by uploading their own palmprint images and observing the matching results in real-time.

![palmprint-demo.gif](output/palmprint-demo.gif)

_The demo showcases the palmprint recognition system's ability to identify palmprints from the same individual.  By using one of the original palmprint images (that was excluded from the training dataset), the model successfully located the other corresponding palmprint images in the database (from the same person) with similarity scores consistently around 90%._

### How to Run the Code

To run this project, navigate to the project directory, follow these steps:

1.  **Install Libraries:** 
    ```bash
    pip install -r requirements.txt
    ```
2. **Run the Streamlit UI:**

    ```bash
    streamlit run app.py
    ```

3. **Access the UI:**  Open your web browser and go to the address provided by Streamlit (usually `http://localhost:8501`).

### Key Libraries Used

*   **PyTorch:** Deep learning framework for model building and training.
*   **torchvision:** PyTorch's library for computer vision tasks, including pre-trained models and image transformations.
*   **OpenCV (cv2):**  Image processing library for preprocessing steps.
*   **NumPy:** Numerical computing library for array operations and data handling.
*   **Streamlit:** Python framework for building interactive web UIs.
*   **scikit-learn (sklearn):** Machine learning library, used for cosine similarity calculation.
*   **PIL (Pillow):** Python Imaging Library for image handling (used via Streamlit).

---
```text
@article{wang2024ai,
  author = {Wang, C. and Wu, X. and Liu, J.},
  title = {A Review on Recent Advances in AI-Driven Image Processing},
  journal = {Applied Sciences},
  volume = {14},
  number = {1},
  pages = {153},
  year = {2024},
  doi = {10.3390/app14010153},
  url = {https://www.mdpi.com/2076-3417/14/1/153}
}
```

_Lucy Tran_
_2025-02-06_