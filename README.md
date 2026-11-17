# Beyond Where, What Matters? Quantifying Visual Cue Contributions with XCues

**Anonymous Repository for CVPR 2027 Double-Blind Review**

This repository contains the official, anonymized implementation of **XCues**, a post-hoc, model-agnostic framework for quantifying visual cue contributions to individual predictions of pretrained image classifiers.

Unlike spatial attribution methods (e.g., Grad-CAM) that highlight *where* evidence is located, XCues quantifies *what* visual properties matter. It evaluates a complete combinatorial intervention space ($2^4 = 16$ states) over four interpretable cue families: **Geometry, Texture, Color, and Context**, yielding mathematically exact Shapley attributions without requiring gradient access or model retraining.

## 🖼️ XCues vs. Grad-CAM
![XCues vs Grad-CAM Comparison](dddffdff.jpeg)

## 🚀 Quickstart: Explain a Single Image

You can generate the exact 16-state Shapley attribution and the comparison plot for any arbitrary image in seconds.

**1. Install Dependencies**
```bash
pip install -r requirements.txt
