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
