# ============================================================
# Notebook 01: Dataset Preparation
# ============================================================
# Prepares the X-ray dataset for training:
# 1. Download/organize datasets
# 2. Compute property annotations
# 3. Generate TIP augmentations
# 4. Create train/val/test splits
# 5. Save everything to Google Drive
# ============================================================

# === Cell 1: Setup ===
# !git clone https://github.com/YOUR_USERNAME/Object-Detection-in-Luggage-Scanner-for-Security.git
# %cd Object-Detection-in-Luggage-Scanner-for-Security
# !pip install -r requirements.txt

import sys
sys.path.insert(0, '.')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.dataset.download import (
    setup_data_directories, list_available_datasets,
    setup_colab_environment, organize_sixray, organize_opixray,
    validate_dataset,
)
from src.dataset.property_annotator import PropertyAnnotator, batch_annotate
from src.dataset.tip_augmentation import TIPAugmenter, labels_to_yolo
from src.dataset.splits import create_splits, create_yolo_data_yaml

config = load_config("configs/default.yaml")

# === Cell 2: Mount Drive & Create Directories ===
# from google.colab import drive
# drive.mount('/content/drive')

# For Colab: use Drive-based paths
# DRIVE_ROOT = "/content/drive/MyDrive/xray_detection"
# For local testing:
DRIVE_ROOT = "data"

data_root = os.path.join(DRIVE_ROOT, "data") if DRIVE_ROOT != "data" else "data"
dirs = setup_data_directories(data_root)

# === Cell 3: List Available Datasets ===
list_available_datasets()

# === Cell 4: Download Dataset ===
# Option A: Download SIXray from Kaggle
# !pip install kaggle
# !mkdir -p ~/.kaggle
# !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d <sixray-dataset-slug> -p data/raw/sixray --unzip

# Option B: Download from GitHub/Google Drive
# For SIXray10 subset:
# !gdown <your-gdrive-file-id> -O data/raw/sixray/SIXray10.zip
# !unzip data/raw/sixray/SIXray10.zip -d data/raw/sixray/

# For OPIXray:
# !git clone https://github.com/OPIXray-author/OPIXray.git data/raw/opixray_repo
# !cp -r data/raw/opixray_repo/OPIXray/* data/raw/opixray/

print("⚠ Please download datasets and place them in:")
print(f"  SIXray: {os.path.join(data_root, 'raw', 'sixray')}")
print(f"  OPIXray: {os.path.join(data_root, 'raw', 'opixray')}")

# === Cell 5: Organize Datasets ===
processed_dir = os.path.join(data_root, "processed")

# Organize SIXray
raw_sixray = os.path.join(data_root, "raw", "sixray")
if os.path.exists(raw_sixray) and os.listdir(raw_sixray):
    organize_sixray(raw_sixray, processed_dir)
else:
    print(f"⚠ SIXray not found. Skipping.")

# Organize OPIXray
raw_opixray = os.path.join(data_root, "raw", "opixray")
if os.path.exists(raw_opixray) and os.listdir(raw_opixray):
    organize_opixray(raw_opixray, processed_dir)
else:
    print(f"⚠ OPIXray not found. Skipping.")

# Validate
stats = validate_dataset(processed_dir)

# === Cell 6: Visualize Sample Images ===
img_dir = os.path.join(processed_dir, "images")
if os.path.exists(img_dir):
    sample_images = sorted(os.listdir(img_dir))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, img_name in zip(axes.flatten(), sample_images):
        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(img_name, fontsize=8)
        ax.axis("off")
    plt.suptitle("Sample X-Ray Images", fontsize=14)
    plt.tight_layout()
    plt.show()

# === Cell 7: Compute Property Annotations ===
print("\n🔬 Computing property annotations...")
annotator = PropertyAnnotator()

img_dir = os.path.join(processed_dir, "images")
label_dir = os.path.join(processed_dir, "labels")

if os.path.exists(img_dir) and os.path.exists(label_dir):
    import glob
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    label_files = [
        os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + ".txt")
        for f in image_files
    ]

    output_csv = os.path.join(data_root, "annotations", "properties.csv")
    props_df = batch_annotate(image_files, label_files, output_csv, annotator)

    print(f"\nProperty annotation stats:")
    print(props_df.describe())
else:
    print(f"⚠ No images/labels for annotation. Creating dummy data for demo...")
    # Create dummy annotation data for pipeline testing
    import pandas as pd
    dummy_records = []
    for i in range(100):
        record = {
            "image_path": f"dummy_{i:04d}.jpg",
            "class_id": np.random.randint(0, 5),
            "bbox_x1": 50, "bbox_y1": 50, "bbox_x2": 200, "bbox_y2": 200,
            "edge_sharpness": np.random.uniform(0, 1),
            "length_to_width_ratio": np.random.uniform(0.5, 5),
            "symmetry_score": np.random.uniform(0, 1),
            "curvature_index": np.random.uniform(0, 1),
            "approximate_volume": np.random.uniform(0, 1),
            "material_category": np.random.randint(0, 4),
            "avg_absorption_intensity": np.random.uniform(0, 1),
            "material_homogeneity": np.random.uniform(0, 1),
            "density_level": np.random.uniform(0, 1),
            "sharp_edge_count": np.random.randint(0, 20),
            "occlusion_score": np.random.uniform(0, 0.5),
        }
        dummy_records.append(record)
    props_df = pd.DataFrame(dummy_records)
    output_csv = os.path.join(data_root, "annotations", "properties.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    props_df.to_csv(output_csv, index=False)
    print(f"✓ Created dummy property annotations: {output_csv}")

# === Cell 8: Visualize Property Distributions ===
from src.utils.visualization import plot_property_distributions
from src.dataset.property_annotator import PROPERTY_NAMES

if 'props_df' in dir() and len(props_df) > 0:
    prop_cols = [c for c in PROPERTY_NAMES if c in props_df.columns]
    if prop_cols:
        prop_values = props_df[prop_cols].values
        class_labels = props_df["class_id"].values if "class_id" in props_df.columns else None
        class_names = config.get("dataset", {}).get("threat_classes", None)

        fig = plot_property_distributions(
            prop_values, prop_cols,
            class_labels=class_labels,
            class_names=class_names,
        )
        plt.show()

# === Cell 9: Create Train/Val/Test Splits ===
print("\n✂ Creating stratified splits...")
img_dir_check = os.path.join(processed_dir, "images")
label_dir_check = os.path.join(processed_dir, "labels")
split_dir = os.path.join(data_root, "splits")

if os.path.exists(img_dir_check) and os.listdir(img_dir_check):
    train_files, val_files, test_files = create_splits(
        img_dir_check, label_dir_check, split_dir,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        create_symlinks=True,
    )

    # Create YOLO data.yaml
    class_names = config.get("dataset", {}).get("threat_classes", ["gun", "knife", "wrench", "pliers", "scissors"])
    create_yolo_data_yaml(split_dir, class_names, os.path.join(data_root, "data.yaml"))
else:
    print("⚠ No images available for splitting.")

# === Cell 10: Summary ===
print(f"\n{'='*60}")
print(f"  ✓ Dataset preparation complete!")
print(f"{'='*60}")
print(f"\nNext step: Run notebook 02_preprocessing_demo.py")
