"""
Dataset Preparation Script.

Automates the full dataset preparation pipeline:
1. Set up directory structure
2. Organize raw datasets
3. Compute property annotations
4. Generate train/val/test splits
5. Create YOLO data.yaml

Usage:
    python scripts/prepare_dataset.py \
        --data-root data \
        --dataset sixray \
        --config configs/default.yaml
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.download import setup_data_directories, organize_sixray, organize_opixray, validate_dataset
from src.dataset.property_annotator import PropertyAnnotator, batch_annotate
from src.dataset.splits import create_splits, create_yolo_data_yaml
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Prepare X-Ray Dataset")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--dataset", default="sixray", choices=["sixray", "opixray", "both"])
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip property annotation")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  X-Ray Dataset Preparation Pipeline")
    print(f"{'='*60}")

    # Step 1: Setup directories
    print(f"\n📁 Step 1: Setting up directories...")
    dirs = setup_data_directories(args.data_root)

    # Step 2: Organize raw data
    print(f"\n📂 Step 2: Organizing raw datasets...")
    processed_dir = os.path.join(args.data_root, "processed")

    if args.dataset in ["sixray", "both"]:
        raw_sixray = os.path.join(args.data_root, "raw", "sixray")
        if os.path.exists(raw_sixray):
            organize_sixray(raw_sixray, processed_dir)
        else:
            print(f"  ⚠ SIXray not found at {raw_sixray}")
            print(f"    Download from: https://github.com/MeioJane/SIXray")

    if args.dataset in ["opixray", "both"]:
        raw_opixray = os.path.join(args.data_root, "raw", "opixray")
        if os.path.exists(raw_opixray):
            organize_opixray(raw_opixray, processed_dir)
        else:
            print(f"  ⚠ OPIXray not found at {raw_opixray}")
            print(f"    Download from: https://github.com/OPIXray-author/OPIXray")

    # Validate
    validate_dataset(processed_dir)

    # Step 3: Property annotation
    if not args.skip_annotation:
        print(f"\n🔬 Step 3: Computing property annotations...")
        img_dir = os.path.join(processed_dir, "images")
        label_dir = os.path.join(processed_dir, "labels")

        if os.path.exists(img_dir) and os.path.exists(label_dir):
            import glob
            image_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            label_files = [
                os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + ".txt")
                for f in image_files
            ]

            output_csv = os.path.join(args.data_root, "annotations", "properties.csv")
            batch_annotate(image_files, label_files, output_csv)
        else:
            print(f"  ⚠ No images/labels found for annotation")

    # Step 4: Create splits
    print(f"\n✂ Step 4: Creating train/val/test splits...")
    img_dir = os.path.join(processed_dir, "images")
    label_dir = os.path.join(processed_dir, "labels")
    split_dir = os.path.join(args.data_root, "splits")

    if os.path.exists(img_dir):
        train_ratio = config.get("dataset", {}).get("train_ratio", 0.70)
        val_ratio = config.get("dataset", {}).get("val_ratio", 0.15)
        test_ratio = config.get("dataset", {}).get("test_ratio", 0.15)

        create_splits(
            img_dir, label_dir, split_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            create_symlinks=True,
        )

        # Step 5: Create YOLO data.yaml
        print(f"\n📝 Step 5: Creating YOLO data.yaml...")
        class_names = config.get("dataset", {}).get("threat_classes", ["threat"])
        create_yolo_data_yaml(
            split_dir, class_names,
            os.path.join(args.data_root, "data.yaml"),
        )
    else:
        print(f"  ⚠ No images found for splitting")

    print(f"\n{'='*60}")
    print(f"  ✓ Dataset preparation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
