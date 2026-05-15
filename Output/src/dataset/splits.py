"""
Dataset splitting utilities.
Generates train/val/test splits with class-balanced stratification.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from collections import Counter


def create_splits(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    create_symlinks: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create stratified train/val/test splits.

    Stratification is based on the class labels in annotation files.
    Images without annotations are placed in the non-threat class.

    Args:
        image_dir: Path to images directory.
        label_dir: Path to YOLO-format labels directory.
        output_dir: Directory to save split files (train.txt, val.txt, test.txt).
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed.
        create_symlinks: If True, create directory structure with symlinks.

    Returns:
        Tuple of (train_files, val_files, test_files) - lists of image filenames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    rng = np.random.RandomState(seed)

    # Collect all images
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if len(image_files) == 0:
        print("⚠ No images found in", image_dir)
        return [], [], []

    # Determine primary class for each image (for stratification)
    image_classes = {}
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path) as f:
                lines = f.readlines()
            if lines:
                # Use the first object's class as the stratification key
                first_class = int(lines[0].strip().split()[0])
                image_classes[img_file] = first_class
            else:
                image_classes[img_file] = -1  # Empty annotation = non-threat
        else:
            image_classes[img_file] = -1  # No annotation = non-threat

    # Group by class
    class_to_images = {}
    for img, cls in image_classes.items():
        class_to_images.setdefault(cls, []).append(img)

    # Stratified split
    train_files, val_files, test_files = [], [], []

    for cls, imgs in class_to_images.items():
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        # Remaining go to test
        train_files.extend(imgs[:n_train])
        val_files.extend(imgs[n_train : n_train + n_val])
        test_files.extend(imgs[n_train + n_val :])

    # Shuffle within splits
    rng.shuffle(train_files)
    rng.shuffle(val_files)
    rng.shuffle(test_files)

    # Save split files
    os.makedirs(output_dir, exist_ok=True)
    _save_split_file(os.path.join(output_dir, "train.txt"), train_files, image_dir)
    _save_split_file(os.path.join(output_dir, "val.txt"), val_files, image_dir)
    _save_split_file(os.path.join(output_dir, "test.txt"), test_files, image_dir)

    # Print statistics
    print(f"\n📊 Dataset Split Statistics:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Train: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"   Val:   {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")
    print(f"   Test:  {len(test_files)} ({len(test_files)/len(image_files)*100:.1f}%)")

    # Class distribution per split
    print(f"\n   Class distribution:")
    for split_name, split_files in [("Train", train_files), ("Val", val_files), ("Test", test_files)]:
        class_counts = Counter(image_classes[f] for f in split_files)
        print(f"   {split_name}: {dict(class_counts)}")

    # Create symlink-based directory structure if requested
    if create_symlinks:
        _create_split_directories(
            image_dir, label_dir, output_dir,
            train_files, val_files, test_files,
        )

    return train_files, val_files, test_files


def _save_split_file(filepath: str, filenames: List[str], image_dir: str):
    """Save split file with full paths."""
    with open(filepath, "w") as f:
        for name in filenames:
            f.write(os.path.join(image_dir, name) + "\n")
    print(f"  ✓ Saved: {filepath} ({len(filenames)} images)")


def _create_split_directories(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
):
    """Create YOLO-style directory structure with copies."""
    import shutil

    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_img_dir = os.path.join(output_dir, split_name, "images")
        split_label_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_label_dir, exist_ok=True)

        for img_file in split_files:
            # Copy image
            src_img = os.path.join(image_dir, img_file)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(split_img_dir, img_file))

            # Copy label
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(split_label_dir, label_file))

    print(f"  ✓ Created split directories in {output_dir}")


def generate_manifest_csv(
    split_dir: str,
    property_csv: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a master manifest CSV combining splits and property annotations.

    Args:
        split_dir: Directory containing train.txt, val.txt, test.txt.
        property_csv: Optional path to property annotations CSV.
        output_path: Path to save the manifest CSV.

    Returns:
        DataFrame with the complete manifest.
    """
    records = []

    for split in ["train", "val", "test"]:
        split_file = os.path.join(split_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            for line in f:
                path = line.strip()
                if path:
                    records.append({"image_path": path, "split": split})

    df = pd.DataFrame(records)

    if property_csv and os.path.exists(property_csv):
        props_df = pd.read_csv(property_csv)
        df = df.merge(props_df, on="image_path", how="left")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✓ Manifest saved: {output_path} ({len(df)} entries)")

    return df


def create_yolo_data_yaml(
    data_dir: str,
    class_names: List[str],
    output_path: str,
):
    """
    Create a data.yaml file for YOLOv8 training.

    Args:
        data_dir: Root directory with train/val/test subdirectories.
        class_names: List of class name strings.
        output_path: Path to save data.yaml.
    """
    import yaml

    data_config = {
        "path": os.path.abspath(data_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(output_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"✓ Created YOLO data.yaml: {output_path}")
