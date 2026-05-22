<div align="center">

# 🛡️ X-Ray Sentry: Dual-Model Deep Property Analyzer

**An Enterprise-Grade, Physics-Informed Threat Detection System for Aviation X-Ray Security**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A three-stage machine learning pipeline combining YOLOv8 object detection, deterministic OpenCV physics, and Random Forest classification — served through a real-time FastAPI backend and an Apple Liquid-Glass inspired React dashboard.*

---

</div>

## 📋 Table of Contents

- [Abstract](#-abstract)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [The Dual-Head ML Pipeline](#-the-dual-head-ml-pipeline)
  - [Stage 1: Vision — PropertyYOLO](#stage-1-vision--propertyyolo)
  - [Stage 2: Deterministic Physics — OpenCV Overrides](#stage-2-deterministic-physics--opencv-overrides)
  - [Stage 3: Classification — Random Forest Ensemble](#stage-3-classification--random-forest-ensemble)
- [The TIP Sandbox](#-the-tip-sandbox-threat-image-projection)
- [Frontend Application](#-frontend-application)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Model Weights](#-model-weights)
- [Acknowledgements](#-acknowledgements)

---

## 📄 Abstract

X-Ray Sentry implements a **three-stage, physics-informed threat detection pipeline** designed for automated baggage screening in aviation security. The system addresses a critical limitation of conventional single-model detectors: their inability to reason about the *physical material properties* of detected objects.

Our architecture couples a **YOLOv8m detection backbone** with a custom **ROI-Aligned Property Regression Head** that extracts an 11-dimensional deep physical property vector per detected object. To mitigate gradient collapse in under-represented property dimensions, we employ **deterministic OpenCV mathematical overrides** — computing absorption intensity, material homogeneity, volumetric fill ratio, and sharp edge count directly from pixel statistics. A downstream **Random Forest ensemble** ingests this enriched feature vector to produce the final threat classification and confidence score.

The system also implements a **Threat Image Projection (TIP) Sandbox** — a real-world aviation security protocol for synthetic data generation — using **Beer-Lambert attenuation physics** with strict alpha-masked blending to project isolated threat objects onto clean baggage X-rays without bounding-box artifacts.

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENT (React 18 + Vite)                      │
│  ┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────────┐    │
│  │ ScannerTab │ │ ConveyorTab  │ │  TIP Tab   │ │ AnalyticsTab  │    │
│  │ (Manual)   │ │ (Live Feed)  │ │ (Sandbox)  │ │ (Recharts)    │    │
│  └─────┬──────┘ └──────┬───────┘ └─────┬──────┘ └───────┬───────┘    │
│        │               │               │                │            │
│        └───────────────┼───────────────┼────────────────┘            │
│                        │  HTTP / JSON  │                             │
├────────────────────────┼───────────────┼─────────────────────────────┤
│                   FASTAPI BACKEND (Uvicorn)                          │
│  ┌─────────────────────┴───────────────┴─────────────────────────┐   │
│  │                    routers/inference.py                        │   │
│  │         POST /api/scan  ·  GET /api/feed  ·  GET /api/stats   │   │
│  └───────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│  ┌───────────────────────────┴───────────────────────────────────┐   │
│  │                            api.py                             │   │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐    │   │
│  │  │  Stage 1: YOLO  │→ │  Stage 2: CV │→ │ Stage 3: RF    │    │   │
│  │  │  PropertyYOLO   │  │  Physics     │  │ Classifier     │    │   │
│  │  │  (best.pt)      │  │  Overrides   │  │ (model2.joblib)│    │   │
│  │  └─────────────────┘  └──────────────┘  └────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                    tip_projector.py                            │   │
│  │  Beer-Lambert TIP  ·  Alpha Masking  ·  Affine Transforms     │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Deep Learning** | PyTorch 2.0+, Ultralytics YOLOv8 | Object detection backbone & property regression |
| **Computer Vision** | OpenCV 4.8+ | Deterministic physics overrides, TIP projection, image I/O |
| **Classical ML** | Scikit-Learn, XGBoost | Random Forest threat classification (Stage 3) |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy | REST API, async inference, scan history persistence |
| **Frontend** | React 18, Vite 5 | Single-page application with real-time state management |
| **Visualization** | Recharts, Lucide-React | Analytics dashboard charts & icon system |
| **Styling** | Vanilla CSS | Custom "Apple Liquid-Glass" glassmorphism design system |
| **Cross-Platform** | `os.path`, `__file__` resolution | Fully OS-agnostic relative pathing (Windows + macOS + Linux) |

---

## 📁 Project Structure

```text
X-Ray-Sentry/
│
├── backend/                          # FastAPI application server
│   ├── main.py                       # App entrypoint, CORS, router registration
│   ├── api.py                        # XRayAPI class: 3-stage ML pipeline
│   ├── database.py                   # SQLAlchemy ORM (ScanLog model, SQLite)
│   ├── tip_projector.py              # BeerLambertTIPProjector engine
│   ├── requirements.txt              # Backend Python dependencies
│   ├── routers/
│   │   ├── inference.py              # /api/scan, /api/feed, /api/model-status
│   │   ├── analytics.py             # /api/mock-analytics, /api/stats, /api/scan-logs
│   │   └── tip.py                    # /api/tip/project endpoint
│   ├── assets/dataset/safe/          # Safe luggage X-ray images (TIP backgrounds)
│   └── temp/                         # Transient inference scratch files
│
├── frontend/                         # React 18 + Vite SPA
│   ├── src/
│   │   ├── App.jsx                   # Root component, state management, API handlers
│   │   ├── index.css                 # Liquid-Glass design system tokens & utilities
│   │   └── components/
│   │       ├── ScannerTab.jsx        # Manual scan UI, Deep Property Analyzer, PDF export
│   │       ├── ConveyorTab.jsx       # Live conveyor belt simulation
│   │       ├── TIPTab.jsx            # Threat Image Projection sandbox controls
│   │       ├── AnalyticsTab.jsx      # Recharts model performance dashboard
│   │       ├── Sidebar.jsx           # Navigation, system status, threat composition
│   │       └── AlertLog.jsx          # Real-time scan history feed
│   └── package.json
│
├── src/                              # Core ML research code
│   ├── model1/
│   │   ├── architecture.py           # PropertyYOLO: YOLOv8m + ROI-Aligned regression head
│   │   ├── train.py                  # Two-stage training loop (detection → properties)
│   │   ├── loss.py                   # Multi-task loss: detection + property regression
│   │   └── evaluate.py              # mAP@50, property MAE, confusion matrices
│   ├── model2/
│   │   ├── prepare_dataset.py        # Feature extraction from PropertyYOLO outputs
│   │   └── train.py                  # Random Forest / XGBoost training & evaluation
│   ├── dataset/                      # Dataset loaders & augmentation pipelines
│   ├── preprocessing/                # X-ray preprocessing utilities
│   └── utils/                        # Logging, config, visualization helpers
│
├── checkpoints/                      # Trained model weights (git-ignored)
│   ├── best.pt                       # Stage 1: YOLOv8m + PropertyYOLO backbone (~50MB)
│   ├── stage2_ultimate.pth           # Stage 2: Custom physics regression head (~155MB)
│   └── model2.joblib                 # Stage 3: Random Forest classifier (~13MB)
│
├── dataset/                          # YOLO-format dataset
│   └── splits/test/images/           # Test split X-rays (used by TIP & Conveyor)
│
├── configs/                          # Training hyperparameter YAML configs
├── notebooks/                        # Jupyter experiment notebooks
├── results/                          # Training logs, evaluation outputs
├── start_scanner.bat                 # 🪟 Windows one-click launcher
├── start.sh                          # 🍎 macOS/Linux launcher
└── requirements.txt                  # Global Python dependencies
```

---

## 🧠 The Dual-Head ML Pipeline

The inference pipeline in [`api.py`](backend/api.py) orchestrates three sequential stages to transform a raw X-ray image into a structured threat assessment.

### Stage 1: Vision — PropertyYOLO

The [`PropertyYOLO`](src/model1/architecture.py) architecture extends a standard **YOLOv8m** detection backbone with a custom **ROI-Aligned Property Regression Head**.

```text
Input X-Ray Image (640×640×3)
        │
        ▼
┌─────────────────────┐
│   YOLOv8m Backbone   │ ──→ Bounding Boxes [x1, y1, x2, y2]
│   (best.pt weights)  │ ──→ Class IDs (gun, knife, wrench, pliers, scissors)
└─────────┬───────────┘ ──→ Confidence Scores
          │
          ▼
┌─────────────────────────────┐
│  ROI-Aligned Feature Crop    │  Crops backbone feature maps per detection
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  Property Regression Head    │  11-dimensional output vector per object:
│  (stage2_ultimate.pth)       │
│                              │  [0] density_level        [6] approx_volume
│  FC(channels → 256) → ReLU  │  [1] edge_sharpness       [7] absorption_intensity
│  FC(256 → 128) → ReLU       │  [2] symmetry_score       [8] material_homogeneity
│  FC(128 → 11)               │  [3] length_width_ratio   [9] sharp_edge_count
│                              │  [4] curvature_index      [10] occlusion_score
│                              │  [5] (reserved)
└──────────────────────────────┘
```

**Training Protocol:**
- **Stage 1** (Epochs 1–30): Freeze property head, train YOLOv8m detection on X-ray dataset.
- **Stage 2** (Epochs 31–60): Freeze YOLO backbone, train property regression head using multi-task loss (detection + property MAE).

---

### Stage 2: Deterministic Physics — OpenCV Overrides

A critical observation during development: indices `[6]–[9]` of the property regression head suffered **gradient collapse** due to insufficient ground-truth diversity. Rather than hallucinate unreliable values, we replace these four dimensions with **deterministic mathematical computations** directly on the raw pixel data using OpenCV.

The function [`calculate_cv2_properties(image, bbox)`](backend/api.py) implements the following:

| Property | Formula | Normalization |
|---|---|---|
| **Absorption Intensity** | `1.0 − (mean(grayscale_crop) / 255.0)` | `[0.0, 1.0]` float — darker pixels → higher absorption → denser material |
| **Material Homogeneity** | `1.0 − (std(grayscale_crop) / 128.0)` | `[0.0, 1.0]` float — lower deviation → more uniform internal structure |
| **Approx. Volume** | `count(pixels < 245) / total_pixels` | `[0.0, 1.0]` float — fill ratio of non-background pixels within the crop |
| **Sharp Edge Count** | `len(cv2.findContours(cv2.Canny(crop, 50, 150)))` | Integer — distinct edge contours, discriminates bladed weapons |

These overrides are **fully deterministic** — identical inputs will always produce identical outputs — eliminating the stochastic noise that the regression head introduced for these dimensions.

---

### Stage 3: Classification — Random Forest Ensemble

The enriched 11-dimensional property vector is passed to a **Random Forest classifier** ([`model2.joblib`](checkpoints/)) trained on the full training split. The model outputs:

- **Threat Classification**: `SAFE` or `THREAT` with per-class breakdown.
- **AI Confidence Score**: Ensemble vote probability.
- **Feature Importance Ranking**: Density Level (28%), Sharp Edge Count (19%), Material Category (16%), Edge Sharpness (11%), and the remaining seven properties.

The Random Forest was selected over a simple threshold-based classifier because it inherently handles the **non-linear interactions** between density, edge count, and material category that characterize real-world weapon signatures.

---

## 🔬 The TIP Sandbox (Threat Image Projection)

Threat Image Projection is a **real-world aviation security protocol** (mandated by ECAC and TSA) used to synthetically inject threat items into clean baggage X-rays to test operator vigilance and AI robustness.

Our implementation in [`tip_projector.py`](backend/tip_projector.py) uses the **Beer-Lambert Law of X-Ray Attenuation**:

```text
I_final = I_background × (I_threat_attenuated / 255)
```

Where attenuation is modulated by a user-controlled thickness parameter:

```text
I_attenuated = 255 − (255 − I_threat) × thickness
```

### Strict Alpha-Masking Pipeline

The projection engine employs a six-step pipeline to prevent the opaque rectangular artifacts that plague naive image overlays:

| Step | Operation | Detail |
|---|---|---|
| **1. Threshold** | `cv2.threshold()` | Adaptive white/black background detection via corner intensity analysis. Creates binary mask *before* any transforms. |
| **2. Morphological Cleanup** | `cv2.erode()` → `cv2.dilate()` | Elliptical kernel removes single-pixel edge noise from the mask. |
| **3. Crop** | `cv2.boundingRect(np.vstack(contours))` | Tight crop using the *union bounding rect* of all contours (handles multi-part objects like scissors). |
| **4. Resize** | `cv2.INTER_NEAREST` for mask | Prevents gray interpolation artifacts — mask stays strictly binary. Re-thresholded at `127` after resize. |
| **5. Rotate** | `cv2.warpAffine()` | Threat padded with neutral gray `(128, 128, 128)` (masked out). Mask padded with black `(0)`. Re-thresholded after rotation. |
| **6. Blend** | `np.where(alpha > 0.5, beer_lambert, background)` | **Hard gate**: background pixels where mask is inactive are *completely untouched*. No partial transparency. |

### Dynamic Dataset Loading

The TIP Sandbox and Conveyor Feed use `os.listdir()` with `random.choice()` to dynamically pull random X-ray images from the local YOLO test split at `dataset/splits/test/images/`. File filtering supports `.jpg`, `.jpeg`, and `.png` extensions. If the directory is empty, a graceful blank-image fallback is served.

---

## 🖥️ Frontend Application

The React 18 frontend implements a premium **Apple Liquid-Glass** design system with glassmorphism panels, micro-animations, and a dark-mode-first color palette.

### Scanner Tab (Manual Analysis)

- Drag-and-drop X-ray upload with real-time inference.
- **Deep Property Analyzer**: Maps the 11-dimensional physics vector to animated progress bars.
- Interactive bounding box overlays with per-object detail expansion.
- **PDF Incident Report Generation**: One-click professional threat assessment export.

### Conveyor Tab (Live Simulation)

- Asynchronous simulation loop mimicking a real airport conveyor belt.
- Bags are queued every **3 seconds** (when playing) with `Pending` status.
- Each bag is sent to the backend for real AI inference; the queue updates asynchronously once the JSON response resolves.
- Maximum queue depth of 10 bags with FIFO eviction.

### TIP Sandbox Tab

- Physics parameter sliders: **Scale**, **Angle**, **Position X/Y**, **Material Thickness** (Beer-Lambert).
- Read-only dataset badges indicating random selection from the local YOLO test split.
- Real-time AI verification panel showing the projected composite's threat assessment.

### Analytics Dashboard

- **5 Stat Cards**: Total Scans, mAP@50, Avg Inference Time, Threats Detected, False Positive Rate.
- **7-Day Threat Stacked Area Chart**: Daily threat category breakdown.
- **YOLO Capability Radar Chart**: Per-class detection accuracy (Gun/Knife/Wrench/Pliers/Scissors).
- **Training Loss Convergence**: Dual-stage loss curves (detection → property regression).
- **Random Forest Feature Importance**: Horizontal bar chart of the 11-property importance ranking.

### Sidebar

- System status indicator with live model health.
- **Threat Composition Widget**: Persistent across all tabs, showing real-time material breakdown (Heavy Metal / Light Metal / Fabric-Plastic / Organic) with animated CSS progress bars.

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+ with `pip`
- **Node.js** 18+ with `npm`
- **Git**
- **Model Weights**: Place the three checkpoint files in the `checkpoints/` directory at the project root:
  - `best.pt` — Stage 1 YOLOv8m backbone
  - `stage2_ultimate.pth` — Stage 2 Physics regression head
  - `model2.joblib` — Stage 3 Random Forest classifier

### Option A: One-Click Launch (Windows)

```batch
:: Double-click or run from terminal:
start_scanner.bat
```

This script automatically boots the FastAPI backend, the Vite dev server, and opens `http://localhost:5173` in your default browser.

### Option B: Manual Launch (macOS / Linux / Windows)

**Terminal 1 — Backend:**

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Launch FastAPI
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm install
npm run dev
```

**Open your browser** at [http://localhost:5173](http://localhost:5173).

> **Note:** The Vite dev server proxies all `/api/*` requests to `localhost:8000` automatically.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/scan` | Upload X-ray image → returns full threat assessment JSON |
| `GET` | `/api/model-status` | Returns loaded model metadata and health status |
| `GET` | `/api/feed` | Returns mock luggage queue metadata for the Conveyor tab |
| `GET` | `/api/feed/image/{mock_type}` | Returns a Base64-encoded dataset X-ray for conveyor simulation |
| `POST` | `/api/tip/project` | Runs Beer-Lambert TIP projection with physics parameters |
| `GET` | `/api/mock-analytics` | Returns training curves, feature importance, confusion matrix |
| `GET` | `/api/stats` | Returns dashboard statistics (total scans, threats, throughput) |
| `GET` | `/api/scan-logs` | Fetches recent scan history from the SQLite database |
| `GET` | `/api/health` | Liveness check — returns engine mode (DL vs CV Fallback) |

### Sample Scan Response

```json
{
  "status": "threat_detected",
  "security_protocol": "CRITICAL_LOCKDOWN_INITIATED",
  "threat_count": 1,
  "diagnostics": {
    "background_noise_events": 0,
    "composition_breakdown": {
      "Heavy/Dense Metal": "100%",
      "Light Metal": "0%",
      "Fabric/Plastic": "0%",
      "Organic": "0%"
    }
  },
  "detections": [
    {
      "id": 1,
      "classification": "GUN",
      "threat_severity_index": 92,
      "confidence_score": 0.891,
      "bounding_box": [120.4, 85.2, 410.7, 290.1],
      "physical_properties": {
        "density_level": 0.847,
        "edge_sharpness": 0.634,
        "symmetry_score": 0.412,
        "length_width_ratio": 1.892,
        "curvature_index": 0.223,
        "approx_volume": 0.731,
        "absorption_intensity": 0.689,
        "material_homogeneity": 0.542,
        "sharp_edge_count": 14,
        "occlusion_score": 0.087
      },
      "material_signature": "Heavy/Dense Metal"
    }
  ]
}
```

---

## ⚖️ Model Weights

| File | Size | Description |
|---|---|---|
| `best.pt` | ~50 MB | YOLOv8m detection backbone, trained on 5-class X-ray dataset |
| `stage2_ultimate.pth` | ~155 MB | Custom property regression head (frozen YOLO, trained on property labels) |
| `model2.joblib` | ~13 MB | Scikit-Learn Random Forest, trained on 11-dim property vectors |

> Weights are not included in the repository due to size constraints. Contact the maintainers for access or retrain using the scripts in `src/model1/train.py` and `src/model2/train.py`.

---

## 🏛️ Cross-Platform Path Resolution

All file paths in the backend use **dynamic resolution** anchored to `__file__`:

```python
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))   # e.g., backend/routers/
BACKEND_DIR  = os.path.dirname(CURRENT_DIR)                 # e.g., backend/
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)                 # e.g., project root

# Model weights
stage1_path = os.path.join(PROJECT_ROOT, "checkpoints", "best.pt")

# Dataset directories
SAFE_DIR    = os.path.join(BACKEND_DIR, "assets", "dataset", "safe")
THREATS_DIR = os.path.join(PROJECT_ROOT, "dataset", "splits", "test", "images")
```

**Zero hardcoded absolute paths.** The application deploys identically on Windows, macOS, and Linux without configuration changes.

---

## 🙏 Acknowledgements

- **Ultralytics** for the YOLOv8 framework.
- **ECAC / TSA** TIP protocol specifications for informing the projection sandbox design.
- The open-source X-ray security imaging research community.

---

<div align="center">

*Built with ⚡ by the X-Ray Sentry Research Team*

</div>
