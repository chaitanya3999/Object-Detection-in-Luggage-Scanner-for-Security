# AI/ML-Based Dual-Model Object Detection System for X-Ray Luggage Scanners

## Overview

A two-model property-based detection system for dual-energy X-ray security scanners. Instead of end-to-end class detection, this architecture uses an intermediate **property vector representation**:

- **Model 1** (YOLOv8 + Property Head): Detects objects and extracts a structured 11-dimensional property vector per object
- **Model 2** (Random Forest / XGBoost): Classifies objects as threats based on their property vectors

This decoupled design enables **scalable threat detection** — new threat types can be added by updating only Model 2's property-to-class mapping, without retraining the expensive Model 1 backbone.

## Property Schema (11 dimensions)

| # | Property | Type | Description |
|---|----------|------|-------------|
| 0 | Edge Sharpness | float [0,1] | Canny edge density |
| 1 | Length-to-Width Ratio | float [0,10] | Bounding box aspect ratio |
| 2 | Symmetry Score | float [0,1] | Horizontal pixel distribution |
| 3 | Curvature Index | float [0,1] | Contour vs convex hull |
| 4 | Approximate Volume | float [0,1] | Area × opacity depth |
| 5 | Material Category | int {0-3} | Organic/Metallic/Mixed/Opaque |
| 6 | Avg Absorption | float [0,1] | Mean grayscale intensity |
| 7 | Material Homogeneity | float [0,1] | Inverted absorption std |
| 8 | Density Level | float [0,1] | Inverted weighted intensity |
| 9 | Sharp Edge Count | int [0,20] | Corner detection count |
| 10 | Occlusion Score | float [0,1] | Bbox overlap fraction |

## Project Structure

```
├── configs/default.yaml          # Central configuration
├── src/
│   ├── preprocessing/
│   │   ├── pipeline.py           # 5-step preprocessing pipeline
│   │   ├── augmentation.py       # albumentations pipelines
│   │   └── segmentation.py       # 3-stage segmentation refinement
│   ├── dataset/
│   │   ├── download.py           # Dataset download/organization
│   │   ├── property_annotator.py # Semi-automated property computation
│   │   ├── tip_augmentation.py   # Threat Image Projection
│   │   ├── splits.py             # Train/val/test stratified splits
│   │   └── xray_dataset.py       # PyTorch Dataset class
│   ├── model1/
│   │   ├── architecture.py       # YOLOv8 + Property Regression Head
│   │   ├── loss.py               # Multi-task loss (Uncertainty Weighting)
│   │   ├── train.py              # Two-stage trainer
│   │   └── evaluate.py           # mAP, MAE/RMSE, material accuracy
│   └── utils/
│       ├── config.py             # YAML config loader
│       └── visualization.py      # Plotting utilities
├── notebooks/
│   ├── 00_setup_and_verify.ipynb
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   └── 03_model1_training.ipynb
├── scripts/
│   ├── prepare_dataset.py        # Dataset pipeline automation
│   └── property_extraction_demo.py
├── requirements.txt
└── .gitignore
```

## Quick Start (Google Colab)

1. **Clone the repo** in Colab:
   ```python
   !git clone https://github.com/YOUR_USERNAME/Object-Detection-in-Luggage-Scanner-for-Security.git
   %cd Object-Detection-in-Luggage-Scanner-for-Security
   !pip install -r requirements.txt
   ```

2. **Run notebooks** in order: `00_setup` → `01_dataset` → `02_preprocessing` → `03_training`

3. **Run demo** (after training):
   ```bash
   python scripts/property_extraction_demo.py --image sample.jpg --checkpoint weights/stage2_best.pt
   ```

## Datasets

- **SIXray** (CVPR 2019): 1M+ X-ray images, 6 threat categories — [GitHub](https://github.com/MeioJane/SIXray)
- **OPIXray** (ACM MM 2020): 8,885 images with occluded items — [GitHub](https://github.com/OPIXray-author/OPIXray)
- **HiXray** (AAAI 2022): 45,364 real airport images — [GitHub](https://github.com/DIG-Beihang/XrayDetection)

## Architecture

```
Input X-Ray Image (4ch)
        │
        ▼
┌─────────────────────────┐
│   YOLOv8 Backbone       │
│   (CSPDarknet53 + PANet) │
└────────┬────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────┐
│ Det.   │  │ Property     │
│ Head   │  │ Regression   │
│ (bbox  │  │ Head (MLP    │
│  +cls) │  │ 512→256→10)  │
└────────┘  └──────────────┘
    │              │
    ▼              ▼
 Detections   Property Vectors
                   │
                   ▼
           ┌──────────────┐
           │   Model 2    │
           │ (RF/XGBoost) │
           └──────────────┘
                   │
                   ▼
            Threat Classification
            + Explanation
```

## Training (Two-Stage)

| Stage | Focus | Epochs | Backbone | Property Head |
|-------|-------|--------|----------|---------------|
| 1 | Detection only | 50-100 | Training (COCO init) | Frozen |
| 2 | Property regression | 30-50 | Frozen | Training |

## Key Technical Contributions

- **Uncertainty Weighting** (Kendall 2018) for multi-task loss balancing
- **Beer-Lambert TIP augmentation** for synthetic threat data generation
- **4-channel dual-energy tensor** construction from single-energy data
- **3-stage segmentation refinement** (Saliency → Mask R-CNN → GrabCut)
- **11-dimensional property schema** including occlusion awareness

## License

This project is part of B.Tech Computer Science Engineering research at VIT Pune (2025-2026).
