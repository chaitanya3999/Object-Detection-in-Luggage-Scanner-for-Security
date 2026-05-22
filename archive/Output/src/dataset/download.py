"""
Dataset download and organization utilities.
Handles downloading SIXray and OPIXray datasets.
"""

import os
import zipfile
import tarfile
import shutil
from pathlib import Path


# Dataset information
DATASET_INFO = {
    "sixray": {
        "description": "SIXray Dataset (Miao et al., CVPR 2019)",
        "url": "https://github.com/MeioJane/SIXray",
        "classes": ["gun", "knife", "wrench", "pliers", "scissors"],
        "num_images": "~8,929 positive + subset of negatives",
        "notes": (
            "SIXray is available via GitHub. The full dataset (1M+ images) "
            "requires requesting access. SIXray10 subset is commonly used. "
            "Download from the GitHub repository or use the Kaggle mirror."
        ),
    },
    "opixray": {
        "description": "OPIXray Dataset (Wei et al., ACM MM 2020)",
        "url": "https://github.com/OPIXray-author/OPIXray",
        "classes": ["folding_knife", "straight_knife", "scissor", "utility_knife", "multi_tool"],
        "num_images": 8885,
        "notes": (
            "OPIXray focuses on occluded prohibited items in X-ray images. "
            "Download from the GitHub repository."
        ),
    },
    "hixray": {
        "description": "HiXray Dataset (Tao et al., AAAI 2022)",
        "url": "https://github.com/DIG-Beihang/XrayDetection",
        "classes": [
            "portable_charger_1", "portable_charger_2", "mobile_phone",
            "laptop", "tablet", "cosmetic", "water", "nonmetallic_lighter",
        ],
        "num_images": 45364,
        "notes": (
            "HiXray provides real airport security check images. "
            "Request access from the authors."
        ),
    },
}


def setup_data_directories(data_root: str) -> dict:
    """
    Create the standardized data directory structure.

    Args:
        data_root: Root data directory path.

    Returns:
        Dictionary of created directory paths.
    """
    dirs = {
        "raw": os.path.join(data_root, "raw"),
        "raw_sixray": os.path.join(data_root, "raw", "sixray"),
        "raw_opixray": os.path.join(data_root, "raw", "opixray"),
        "processed": os.path.join(data_root, "processed"),
        "processed_images": os.path.join(data_root, "processed", "images"),
        "processed_labels": os.path.join(data_root, "processed", "labels"),
        "annotations": os.path.join(data_root, "annotations"),
        "annotations_properties": os.path.join(data_root, "annotations", "properties"),
        "augmented": os.path.join(data_root, "augmented"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        print(f"  ✓ Created: {path}")

    return dirs


def list_available_datasets():
    """Print information about available datasets."""
    print("=" * 60)
    print("Available X-Ray Security Datasets")
    print("=" * 60)
    for name, info in DATASET_INFO.items():
        print(f"\n📦 {name.upper()}")
        print(f"   {info['description']}")
        print(f"   URL: {info['url']}")
        print(f"   Classes: {', '.join(info['classes'])}")
        print(f"   Images: {info['num_images']}")
        print(f"   Note: {info['notes']}")
    print()


def download_from_kaggle(dataset_slug: str, download_dir: str):
    """
    Download a dataset from Kaggle (requires kaggle API token).

    Args:
        dataset_slug: Kaggle dataset slug (e.g., 'username/dataset-name').
        download_dir: Directory to download to.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)
        print(f"✓ Downloaded {dataset_slug} to {download_dir}")
    except ImportError:
        print("⚠ Kaggle API not installed. Install with: pip install kaggle")
        print("  Then set up your API token: https://www.kaggle.com/docs/api")
    except Exception as e:
        print(f"⚠ Kaggle download failed: {e}")
        print("  Try manual download from: https://www.kaggle.com/datasets/")


def extract_archive(archive_path: str, extract_dir: str):
    """Extract zip or tar archives."""
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tf:
            tf.extractall(extract_dir)
    print(f"✓ Extracted to {extract_dir}")


def organize_sixray(raw_dir: str, output_dir: str, subset: str = "SIXray10"):
    """
    Organize SIXray dataset into a standardized format.

    Expected raw structure:
        raw_dir/
        ├── SIXray10/  (or SIXray100)
        │   ├── positive/
        │   │   ├── P00001.jpg
        │   │   └── ...
        │   ├── negative/
        │   │   ├── N00001.jpg
        │   │   └── ...
        │   └── annotation/
        │       ├── gun.txt (or xmls)
        │       └── ...

    Standardized output:
        output_dir/
        ├── images/
        │   ├── img_0001.jpg
        │   └── ...
        └── labels/
            ├── img_0001.txt  (YOLO format: class x_center y_center width height)
            └── ...
    """
    img_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    source_dir = os.path.join(raw_dir, subset)
    if not os.path.exists(source_dir):
        print(f"⚠ Source directory not found: {source_dir}")
        print(f"  Please download SIXray and place it in: {raw_dir}")
        return

    # Process positive images
    pos_dir = os.path.join(source_dir, "positive")
    if os.path.exists(pos_dir):
        for img_name in sorted(os.listdir(pos_dir)):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                src = os.path.join(pos_dir, img_name)
                dst = os.path.join(img_dir, img_name)
                shutil.copy2(src, dst)

    # Process negative images
    neg_dir = os.path.join(source_dir, "negative")
    if os.path.exists(neg_dir):
        for img_name in sorted(os.listdir(neg_dir)):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                src = os.path.join(neg_dir, img_name)
                dst = os.path.join(img_dir, img_name)
                shutil.copy2(src, dst)

    total = len(os.listdir(img_dir))
    print(f"✓ Organized {total} images from SIXray {subset}")


def organize_opixray(raw_dir: str, output_dir: str):
    """
    Organize OPIXray dataset into standardized format.

    OPIXray typically comes with:
        raw_dir/
        ├── train/
        │   ├── image/
        │   └── label/
        └── test/
            ├── image/
            └── label/
    """
    img_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for split in ["train", "test"]:
        split_img_dir = os.path.join(raw_dir, split, "image")
        split_label_dir = os.path.join(raw_dir, split, "label")

        if os.path.exists(split_img_dir):
            for img_name in sorted(os.listdir(split_img_dir)):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    src = os.path.join(split_img_dir, img_name)
                    dst = os.path.join(img_dir, f"{split}_{img_name}")
                    shutil.copy2(src, dst)

        if os.path.exists(split_label_dir):
            for label_name in sorted(os.listdir(split_label_dir)):
                if label_name.endswith(".txt"):
                    src = os.path.join(split_label_dir, label_name)
                    dst = os.path.join(label_dir, f"{split}_{label_name}")
                    shutil.copy2(src, dst)

    total = len(os.listdir(img_dir))
    print(f"✓ Organized {total} images from OPIXray")


def validate_dataset(data_dir: str) -> dict:
    """
    Validate a dataset directory and return statistics.

    Args:
        data_dir: Path to organized dataset directory with images/ and labels/.

    Returns:
        Dictionary with dataset statistics.
    """
    img_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")

    stats = {
        "total_images": 0,
        "total_labels": 0,
        "images_without_labels": [],
        "labels_without_images": [],
        "image_extensions": {},
    }

    if os.path.exists(img_dir):
        images = set()
        for f in os.listdir(img_dir):
            ext = os.path.splitext(f)[1].lower()
            stats["image_extensions"][ext] = stats["image_extensions"].get(ext, 0) + 1
            images.add(os.path.splitext(f)[0])
        stats["total_images"] = len(images)

    if os.path.exists(label_dir):
        labels = set()
        for f in os.listdir(label_dir):
            if f.endswith(".txt"):
                labels.add(os.path.splitext(f)[0])
        stats["total_labels"] = len(labels)

        if os.path.exists(img_dir):
            stats["images_without_labels"] = list(images - labels)[:10]
            stats["labels_without_images"] = list(labels - images)[:10]

    print(f"\n📊 Dataset Validation: {data_dir}")
    print(f"   Images: {stats['total_images']}")
    print(f"   Labels: {stats['total_labels']}")
    print(f"   Extensions: {stats['image_extensions']}")
    if stats["images_without_labels"]:
        print(f"   ⚠ {len(stats['images_without_labels'])} images without labels (showing first 10)")
    if stats["labels_without_images"]:
        print(f"   ⚠ {len(stats['labels_without_images'])} labels without images (showing first 10)")

    return stats


# ============================================================
# Colab-specific helper
# ============================================================

def setup_colab_environment(drive_path: str = "/content/drive/MyDrive/xray_detection"):
    """
    Set up the Google Colab environment:
    1. Mount Google Drive
    2. Create directory structure on Drive
    3. Return resolved paths

    Usage in Colab:
        from src.dataset.download import setup_colab_environment
        paths = setup_colab_environment()
    """
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("✓ Google Drive mounted")
    except ImportError:
        print("⚠ Not running in Google Colab")

    dirs = setup_data_directories(os.path.join(drive_path, "data"))
    os.makedirs(os.path.join(drive_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(drive_path, "results"), exist_ok=True)

    return {
        "root": drive_path,
        "data": os.path.join(drive_path, "data"),
        "checkpoints": os.path.join(drive_path, "checkpoints"),
        "results": os.path.join(drive_path, "results"),
        **dirs,
    }
