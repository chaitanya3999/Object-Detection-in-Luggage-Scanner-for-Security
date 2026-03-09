# ============================================================
# Notebook 00: Setup & Verify Environment
# ============================================================
# Run this notebook first in Google Colab to verify all
# dependencies are installed and GPU is available.
#
# Steps:
# 1. Clone repository
# 2. Install dependencies
# 3. Verify imports
# 4. Check GPU availability
# 5. Mount Google Drive
# ============================================================

# === Cell 1: Clone Repository ===
# !git clone https://github.com/YOUR_USERNAME/Object-Detection-in-Luggage-Scanner-for-Security.git
# %cd Object-Detection-in-Luggage-Scanner-for-Security

# === Cell 2: Install Dependencies ===
# !pip install -r requirements.txt
# !pip install 'git+https://github.com/facebookresearch/detectron2.git'

# === Cell 3: Verify Core Imports ===
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

print(f"PyTorch version:     {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"OpenCV version:      {cv2.__version__}")
print(f"NumPy version:       {np.__version__}")
print(f"Pandas version:      {pd.__version__}")

# === Cell 4: Check GPU ===
print(f"\n{'='*40}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠ No GPU detected. Go to Runtime > Change runtime type > GPU")

# === Cell 5: Verify Project Imports ===
import sys
sys.path.insert(0, '.')

from src.preprocessing.pipeline import preprocess_xray, PreprocessingPipeline
from src.dataset.property_annotator import PropertyAnnotator
from src.dataset.download import setup_data_directories, list_available_datasets
from src.utils.config import load_config
from src.utils.visualization import visualize_detections

print("\n✓ All project modules imported successfully!")

# === Cell 6: Test Preprocessing Pipeline ===
dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = preprocess_xray(dummy_image)
print(f"\nPreprocessing test:")
print(f"  Input:  (480, 640, 3) uint8")
print(f"  Output: {result.shape} float32")
assert result.shape == (4, 640, 640), f"Expected (4,640,640), got {result.shape}"
print("  ✓ PASSED")

# === Cell 7: Test Property Annotator ===
annotator = PropertyAnnotator()
dummy_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
dummy_mask = np.ones((480, 640), dtype=np.uint8)
dummy_bbox = [100, 100, 300, 300]
props = annotator.compute_properties(dummy_gray, dummy_mask, dummy_bbox)
assert len(props) == 11, f"Expected 11 properties, got {len(props)}"
print(f"\nProperty annotator test:")
print(f"  Computed {len(props)} properties: ✓ PASSED")
for name, value in props.items():
    print(f"    {name}: {value:.4f}" if isinstance(value, float) else f"    {name}: {value}")

# === Cell 8: Load Config ===
config = load_config("configs/default.yaml")
print(f"\n✓ Config loaded: {config['project']['name']} v{config['project']['version']}")
print(f"  Properties: {config['property_schema']['num_properties']}")
print(f"  Model: {config['model1']['backbone']}")

# === Cell 9: Mount Google Drive ===
# from google.colab import drive
# drive.mount('/content/drive')
# print("✓ Google Drive mounted")

# === Cell 10: Create Directory Structure on Drive ===
# from src.dataset.download import setup_colab_environment
# paths = setup_colab_environment()
# print(f"\n✓ Directories created on Google Drive")
# for key, path in paths.items():
#     print(f"  {key}: {path}")

print(f"\n{'='*60}")
print(f"  ✓ Setup complete! All systems operational.")
print(f"{'='*60}")
