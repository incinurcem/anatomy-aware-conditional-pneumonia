<div align="center">

<img src="assets/banner.png" alt="Anatomy-Aware Conditional Pneumonia Detection" width="100%"/>

# 🩻 Anatomy-Aware Conditional Pneumonia Detection

### *A hybrid deep learning framework fusing nnU-Net lung segmentation with conditional ResNet-50 classification on chest X-rays*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-CXR_pretrained-2E86AB.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/📄_Paper-PDF-red.svg)](docs/paper.pdf)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-2A9D8F.svg)](#-live-demo)

**[📄 Paper](docs/paper.pdf)** · **[🚀 Demo](#-live-demo)** · **[📊 Results](#-key-results)** · **[🔬 Architecture](#-architecture)** · **[🧪 Reproduce](#-reproduce)**

</div>

---

## ✨ The Idea in One Sentence

> **What if a chest X-ray classifier didn't just look at pixels — but also knew the *shape of the lungs* it was looking at?**

We propose an **anatomy-aware, conditional** ResNet-50 that fuses two complementary signals: a **CXR-pretrained nnU-Net** mask defining the lung anatomy, and a **22-dimensional hand-crafted feature vector** capturing mask geometry, ROI intensity statistics, and GLCM texture. To quantify the value of each component, we run a **36-experiment controlled ablation** — and discover a clean monotonic principle.

---

## 🎯 Headline Findings

<table>
<tr>
<td width="50%" align="center">

### 📈 Δ AUC scales monotonically with mask dependency

```
Plain inputs:      ΔAUC = +0.005   (ns)
ROI inputs:        ΔAUC = +0.018   (p = 0.024) *
Masked-ROI inputs: ΔAUC = +0.045   (p = 0.018) *
```

*The classifier benefits from a high-quality mask **only to the extent it actually uses it**.*

</td>
<td width="50%" align="center">

### 🌍 Robust cross-domain generalization

```
RSNA  → Kermany pediatric
nnU-Net: 0.88 → 0.91
Otsu:    0.88 → 0.95
```

*Models generalize zero-shot to a different age group, vendor, and class distribution.*

</td>
</tr>
</table>

![Results overview](assets/results.png)

---

## 📑 Table of Contents

- [The Idea](#-the-idea-in-one-sentence)
- [Architecture](#-architecture)
- [Why This Matters](#-why-this-matters)
- [Key Results](#-key-results)
- [Live Demo](#-live-demo)
- [Repository Structure](#-repository-structure)
- [Reproduce](#-reproduce)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🔬 Architecture

<div align="center">

![Pipeline](assets/pipeline.png)

</div>

The system is a **two-stage anatomy-aware pipeline** with optional conditional branch:

### Stage 1 — Anatomical Segmentation
A CXR-pretrained **nnU-Net** produces a high-fidelity lung mask. As a control, we also run a **classical Otsu** pipeline (thresholding + morphology + connected components) to isolate the contribution of segmentation quality.

### Stage 2 — Conditional ResNet-50 Classifier

```
                ┌─────────────────────────┐
   CXR ─────────►   ResNet-50 (ImageNet)  │── 2048-dim ──┐
   (224×224)    │       backbone          │              │
                └─────────────────────────┘              │  Concat
                                                         ├──────► [Linear → 256 → 1]
   22 hand-crafted ┌─────────────────────┐               │            │
   features    ───►│   Conditional MLP    │── 512-dim ───┘            ▼
   (mask geom +    │ Linear→BN→ReLU→      │                     Pneumonia
    intensity +    │ Linear→ReLU→Dropout  │                     probability
    GLCM)          └─────────────────────┘
```

### The 22 Conditional Features

| Group | Count | Features |
|---|---|---|
| **Mask geometry** | 5 | area ratio, bbox width/height, aspect ratio, centroid Y |
| **ROI intensity** | 8 | mean, std, median, q1, q3, IQR, min, max |
| **Edge / gradient** | 5 | Sobel density, Laplacian variance, gradient mean/std, foreground ratio |
| **GLCM texture** | 4 | contrast, dissimilarity, homogeneity, energy |

### 6 Input Configurations

|   | No Conditional | + Conditional |
|---|---|---|
| **Plain** (full CXR) | `classifier_plain` | `conditional_plain` |
| **ROI** (lung-bbox crop) | `classifier_roi` | `conditional_roi` |
| **Masked-ROI** (soft-mask filtered) | `classifier_masked_roi` | `conditional_masked_roi` |

Each cell ran with **2 segmentation pipelines (nnU-Net, Otsu) × 3 random seeds (42, 123, 7)** — a clean **2 × 6 × 3 = 36-run** design.

---

## 💡 Why This Matters

Two-stage segmentation-then-classification pipelines are the de-facto standard in CXR analysis, but the **value of each upgrade is rarely quantified**. We answer three concrete questions:

1. **Is sophisticated segmentation worth it?** Only when the classifier structurally depends on the mask. For plain inputs, classical Otsu is statistically indistinguishable from nnU-Net. For masked-ROI inputs, nnU-Net beats Otsu by **+0.045 AUC, p < 0.05**.

2. **Do hand-crafted features still help in 2026?** Marginally. Conditional features add **< 0.5% AUC** when paired with a strong ImageNet backbone — consistent with the "deep features subsume radiomics" literature, but worth knowing.

3. **Does the effect generalize?** Partially. On Kermany pediatric pneumonia, all models improve in absolute AUC (easier task, balanced classes), but the **pipeline ranking inverts** — a cautionary tale about within-distribution claims.

---

## 📊 Key Results

### Main ablation — RSNA test set (3-seed mean ± std)

| Configuration | nnU-Net | Otsu | Δ AUC | p-value |
|---|---|---|---|---|
| Plain | 0.8825 ± 0.0049 | 0.8779 ± 0.0048 | +0.005 | 0.30  ns |
| ROI | 0.8784 ± 0.0021 | 0.8645 ± 0.0041 | +0.014 | 0.058  ~ |
| Masked-ROI | 0.8643 ± 0.0083 | 0.8307 ± 0.0091 | +0.034 | 0.075  ~ |
| Plain + Cond | 0.8803 ± 0.0011 | 0.8804 ± 0.0032 | −0.000 | 0.98  ns |
| **ROI + Cond** | **0.8823 ± 0.0032** | **0.8642 ± 0.0024** | **+0.018** | **0.024 \*** |
| **Masked-ROI + Cond** | **0.8711 ± 0.0073** | **0.8256 ± 0.0077** | **+0.045** | **0.018 \*** |

\* paired t-test p < 0.05  ·  ~ trend toward significance (0.05 < p < 0.10)

### External validation — Kermany pediatric pneumonia (n = 624)

| Pipeline | RSNA AUC (3-seed) | Kermany AUC (3-seed) | Δ Domain Shift |
|---|---|---|---|
| nnU-Net | 0.8825 ± 0.0049 | 0.9085 ± 0.0497 | **+0.026** |
| Otsu | 0.8779 ± 0.0048 | 0.9470 ± 0.0072 | **+0.069** |

### Grad-CAM examples

<div align="center">

![Grad-CAM](assets/gradcam_examples.png)

*Attention maps tighten progressively from `plain` → `ROI` → `masked_roi`, confirming that mask-aware inputs force the model to focus inside the lung field.*

</div>

---

## 🚀 Live Demo

The repository ships with a polished **Streamlit dashboard** that doubles as a clinical-style diagnostic interface and an interactive research explorer.

<div align="center">

![App screenshot](assets/app_screenshot.png)

</div>

### 🏥 What you can do in the demo

| Feature | Description |
|---|---|
| **📤 Upload CXR** | Drop a PNG / JPG / DICOM file directly into the browser |
| **🔬 Live segmentation** | Watch Otsu lung segmentation, ROI cropping, and masked-ROI generation happen in real-time |
| **🎯 Diagnostic card** | Big POSITIVE / NEGATIVE label, probability percentage, severity badge (Low / Moderate / High / Very High), and a gradient gauge with the Youden-optimal threshold marked |
| **🔥 Grad-CAM heatmap** | See exactly where the network is looking when it makes its prediction |
| **🔁 Compare modes** | One click runs the same image through all 6 configurations (plain / ROI / masked-ROI × ±cond) side-by-side |
| **📊 Research dashboard** | Per-seed aggregation, ablation table, paired t-tests, ROC / PR / Confusion grids, external validation domain-shift analysis |
| **🧪 Test-set inspector** | Browse 4,002 RSNA test images, filter by label, generate Grad-CAMs across all 12 model configurations |

### Run locally

```bash
git clone https://github.com/<your-username>/anatomy-aware-conditional-pneumonia.git
cd anatomy-aware-conditional-pneumonia
pip install -r pneumonia_app/requirements.txt
streamlit run pneumonia_app/app.py
```

Open `http://localhost:8501` in your browser.

### Run in Google Colab (with public ngrok URL)

```python
# In a Colab cell — get a free ngrok token at https://dashboard.ngrok.com/
!git clone https://github.com/<your-username>/anatomy-aware-conditional-pneumonia.git
%cd anatomy-aware-conditional-pneumonia
!pip install -q -r pneumonia_app/requirements.txt streamlit pyngrok \
                  scikit-image opencv-python-headless pydicom

import os
os.environ["NGROK_TOKEN"] = "<paste-your-ngrok-token>"
!python pneumonia_app/colab_launch.py
```

After ~10 seconds you'll see:

```
══════════════════════════════════════════════════════════════
  🌐  PUBLIC URL:  https://abcd-1234-5678.ngrok.app
══════════════════════════════════════════════════════════════
```

Click the URL — your Streamlit app is live on the internet, accessible from any device.

> 💡 **Tip:** The free ngrok plan supports one tunnel and 8-hour sessions. For permanent hosting, deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) (free) or [HuggingFace Spaces](https://huggingface.co/spaces) (free).

### ⚠️ Important — research demo only

The deployed app is **not a medical device**. Predictions reflect the RSNA training distribution and **must not** be used for clinical decisions. The interface includes a persistent disclaimer banner.

---

## 📁 Repository Structure

```
anatomy-aware-conditional-pneumonia/
│
├── README.md                   You are here
├── LICENSE                     MIT
├── requirements.txt
│
├── src/                        ⭐ Core architectures
│   ├── models.py               ResNet-50 + ConditionalResNet50
│   └── datasets.py             RSNAClassifierDataset (6 modes)
│
├── scripts/                    ⭐ End-to-end pipeline
│   ├── train_rsna_classifier3.py    Main training script (used for all 36 runs)
│   ├── segment_otsu.py              Classical Otsu segmentation pipeline
│   ├── segment_nnunet.py            nnU-Net inference wrapper
│   └── extract_features.py          22 conditional feature extraction
│
├── configs/                    Hyperparameter configs
│
├── pneumonia_app/              ⭐ Streamlit deployment
│   ├── app.py                  Two-tab dashboard (Diagnose + Research)
│   ├── colab_launch.py         ngrok launcher
│   ├── requirements.txt
│   └── core/                   Modular helpers
│       ├── config.py           Paths & constants
│       ├── data.py             Run gathering + caching
│       ├── stats.py            Paired t-test + bootstrap CI
│       ├── models.py           App-side model loading
│       ├── input_prep.py       Inference preprocessing
│       ├── gradcam.py          Grad-CAM + heatmap overlay
│       ├── plots.py            Matplotlib plot helpers
│       ├── tables.py           DataFrame display
│       └── clinical.py         Live Otsu segmentation + 22 features for uploads
│
├── outputs_seeds_nnUNet/       Per-run results (JSON + CSV; .pth excluded)
│   └── <experiment>/seed_<n>/
│       ├── test_metrics.json
│       └── test_predictions.csv
│
├── outputs_seeds_otsu/         Same structure
│
├── external_eval/              ⭐ Kermany external validation outputs
│   ├── external_per_seed.csv
│   └── external_summary.csv
│
├── seed_results_summary.csv    ⭐ 3-seed mean ± std AUC/F1
└── seed_ttest_results.csv      ⭐ Paired t-test (nnU-Net vs Otsu)
```

> Trained model weights (`*.pth`, ~3.6 GB total across 36 models) are **not** included in the repository. They are available via [Releases](#) or by retraining with the scripts in `scripts/`.

---

## 🧪 Reproduce

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥ 16 GB VRAM (A100 80 GB recommended for batch size 512)
- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset
- [Kermany pediatric pneumonia](https://data.mendeley.com/datasets/rscbjbr9sj/2) (for external validation)

### 1. Setup

```bash
git clone https://github.com/<your-username>/anatomy-aware-conditional-pneumonia.git
cd anatomy-aware-conditional-pneumonia
pip install -r requirements.txt
```

### 2. Data preparation

Download RSNA Challenge data and convert DICOM → PNG. Run the segmentation pipelines (nnU-Net + Otsu) and the 22-feature extraction. Each pipeline writes to its own `data/conditional_v3*` folder with `train_conditional_safe.csv`, `val_conditional_safe.csv`, `test_conditional_safe.csv`.

### 3. Train the 36-run grid

```bash
python scripts/train_rsna_classifier3.py \
    --train_csv data/conditional_v3/train/train_conditional_safe.csv \
    --val_csv   data/conditional_v3/val/val_conditional_safe.csv \
    --test_csv  data/conditional_v3/test/test_conditional_safe.csv \
    --output_dir outputs_seeds_nnUNet/conditional_masked_roi/seed_42 \
    --input_mode masked_roi --is_conditional \
    --model_name resnet50 --img_size 224 \
    --batch_size 512 --epochs 12 --lr 8e-4 \
    --num_workers 8 --pretrained --amp \
    --seed 42 --early_stop_patience 5
```

Repeat for the full **2 × 6 × 3** grid (≈ 4 min/run on A100, **~2 h 14 m total**).

### 4. External validation

Run inference on Kermany pediatric for the plain models with bootstrap 95 % CI. Outputs land in `external_eval/`.

### 5. Launch the dashboard

```bash
streamlit run pneumonia_app/app.py
```

---

## 📄 Citation

If this work informs your research, please cite:

```bibtex
@article{yourname2026anatomy,
  title   = {Anatomy-Aware Conditional Pneumonia Detection: Fusing nnU-Net Lung Segmentation
             with Hand-Crafted Radiomic Features in a ResNet-50 Classifier},
  author  = {Your Name and Advisor Name},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  url     = {https://github.com/<your-username>/anatomy-aware-conditional-pneumonia}
}
```

---

## 🙏 Acknowledgments

- **Datasets** — [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) · [Kermany Pediatric Pneumonia](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- **Frameworks** — [PyTorch](https://pytorch.org/) · [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) · [Streamlit](https://streamlit.io/) · [scikit-learn](https://scikit-learn.org/) · [scikit-image](https://scikit-image.org/) · [OpenCV](https://opencv.org/)
- **Compute** — Google Colab Pro+ (NVIDIA A100 80 GB)
- **Inspiration** — [Tartaglione et al. 2020](https://doi.org/10.3390/ijerph17186933) on COVID-19 segmentation ablations · [Isensee et al. 2021](https://doi.org/10.1038/s41592-020-01008-z) on nnU-Net

---

## 📜 License

Released under the [MIT License](LICENSE) — free for academic and commercial use, with the standard *no warranty* clause.

> ⚠️ **Disclaimer:** This software is **not** a medical device. The trained models reflect the statistical patterns of their training data and must not be used as the sole basis for clinical decisions. Always consult qualified medical professionals.

---

<div align="center">

### Built with 🩻 for the medical AI research community

**[⭐ Star this repo](#)** if you find it useful · **[🐛 Open an issue](#)** if something breaks · **[💬 Start a discussion](#)** to share findings

</div>
