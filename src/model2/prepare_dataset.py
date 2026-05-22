"""
Model 2 Tabular Dataset Generator.

Extracts 11-dimensional property vectors from the SIXray / OPIXray datasets.
It processes both:
1. Annotated threat objects (Guns, Knives, etc.) -> Label as threat class.
2. Segmented benign objects (using contour detection on non-overlapping regions) -> Label as safe.

Saves the result as a tabular CSV dataset for training Random Forest / XGBoost models.
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset.property_annotator import PropertyAnnotator
from src.utils.config import load_config


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


def extract_safe_candidates(image, threat_bboxes, min_area=900, max_area_ratio=0.9):
    """
    Extract safe background objects using OpenCV contour detection.
    Ensures candidates do not overlap with ground-truth threat bounding boxes.
    """
    h, w = image.shape[:2]
    # Threshold empty background (usually very light dual-energy background, near 220-255)
    # Binary inverse selects dark regions (objects)
    _, thresh = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)

    # Clean up threshold mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    safe_candidates = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        
        # Filter noise and massive background wrappers
        if area < min_area or area > (h * w * max_area_ratio):
            continue
            
        candidate_bbox = [x, y, x + cw, y + ch]
        
        # Check overlap with any threat bbox
        is_safe = True
        for t_bbox in threat_bboxes:
            if compute_iou(candidate_bbox, t_bbox) > 0.1:
                is_safe = False
                break
                
        if is_safe:
            # Generate binary mask of the contour
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
            safe_candidates.append({
                "bbox": candidate_bbox,
                "mask": mask
            })
            
    return safe_candidates


def main():
    parser = argparse.ArgumentParser(description="Generate Tabular Dataset for Model 2")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--dataset", default="sixray", help="Specific dataset directory under raw")
    parser.add_argument("--output-csv", default="data/annotations/model2_dataset.csv", help="Output CSV path")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--num-images", type=int, default=0, help="Limit number of images to process (0 = all)")
    args = parser.parse_args()

    config = load_config(args.config)
    threat_classes = config.get("dataset", {}).get("threat_classes", ["gun", "knife", "wrench", "pliers", "scissors"])
    
    # Class mapping
    # 0 to 4: threat classes, 5: safe/non-threat
    class_map = {i: name for i, name in enumerate(threat_classes)}
    class_map[len(threat_classes)] = "safe"

    print(f"\n{'='*65}")
    print(f"  Model 2 Tabular Dataset Generator")
    print(f"{'='*65}")
    
    raw_dir = os.path.join(args.data_root, "raw", args.dataset)
    if not os.path.exists(raw_dir):
        print(f"Error: Raw dataset path not found at {raw_dir}")
        sys.exit(1)

    # Collect images and annotations across all splits (train, valid, test)
    image_files = []
    for split in ["train", "valid", "test", "val"]:
        split_img_dir = os.path.join(raw_dir, split, "images")
        if os.path.exists(split_img_dir):
            image_files.extend(glob.glob(os.path.join(split_img_dir, "*.*")))
            
    # Remove duplicates and filter valid image formats
    image_files = sorted(list(set([
        f for f in image_files 
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])))

    if len(image_files) == 0:
        print("No images found in dataset splits!")
        sys.exit(1)

    if args.num_images > 0:
        image_files = image_files[:args.num_images]
        print(f"Processing limited subset of {len(image_files)} images...")
    else:
        print(f"Found {len(image_files)} total images to process...")

    annotator = PropertyAnnotator()
    records = []

    for img_path in tqdm(image_files, desc="Extracting properties"):
        # Load grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        h, w = image.shape[:2]

        # Parse YOLO label path
        # Assuming folder structure split/images/img.jpg and split/labels/img.txt
        img_p = Path(img_path)
        ann_path = os.path.join(img_p.parent.parent, "labels", img_p.stem + ".txt")

        # Parse ground truth threat boxes
        threat_bboxes = []
        threat_classes_in_img = []
        
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        
                        # Convert YOLO coords back to pixel coords
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        
                        # Clip to boundary
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 > x1 and y2 > y1:
                            threat_bboxes.append([x1, y1, x2, y2])
                            threat_classes_in_img.append(cls)

        # 1. Process threat objects
        for i, (bbox, cls) in enumerate(zip(threat_bboxes, threat_classes_in_img)):
            # Create mask for this box
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            
            # Compute properties
            props = annotator.compute_properties(
                image, mask, bbox, all_bboxes=threat_bboxes, current_idx=i
            )
            
            cls_name = threat_classes[cls] if cls < len(threat_classes) else "unknown"
            
            # Assign threat level
            if cls_name in ["gun", "knife"]:
                level = "CRITICAL"
            else:
                level = "WARNING"

            record = {
                "image_path": os.path.abspath(img_path),
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "class_id": cls,
                "label": cls_name,
                "threat_level": level
            }
            record.update(props)
            records.append(record)

        # 2. Process benign/safe background objects
        safe_candidates = extract_safe_candidates(image, threat_bboxes)
        
        # To avoid massive class imbalance, limit safe objects per image
        max_safe_per_img = max(2, len(threat_bboxes) * 2)
        if len(safe_candidates) > max_safe_per_img:
            # Sample subset randomly
            indices = np.random.choice(len(safe_candidates), max_safe_per_img, replace=False)
            safe_candidates = [safe_candidates[idx] for idx in indices]

        all_boxes_with_safe = threat_bboxes + [c["bbox"] for c in safe_candidates]

        for i, cand in enumerate(safe_candidates):
            bbox = cand["bbox"]
            mask = cand["mask"]
            
            # Compute properties
            idx_in_all = len(threat_bboxes) + i
            props = annotator.compute_properties(
                image, mask, bbox, all_bboxes=all_boxes_with_safe, current_idx=idx_in_all
            )
            
            record = {
                "image_path": os.path.abspath(img_path),
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "class_id": len(threat_classes), # Save class id as len(threat_classes)
                "label": "safe",
                "threat_level": "SAFE"
            }
            record.update(props)
            records.append(record)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        df.to_csv(args.output_csv, index=False)
        print(f"\n✓ Extracted properties for {len(df)} objects.")
        print(f"  Threats: {len(df[df['label'] != 'safe'])}")
        print(f"  Safe background objects: {len(df[df['label'] == 'safe'])}")
        print(f"✓ Saved Model 2 training dataset to: {os.path.abspath(args.output_csv)}")
        
        # Display class distribution
        print("\nClass distribution:")
        print(df["label"].value_counts().to_string())
    else:
        print("\n⚠ No records extracted! Check dataset image and annotation paths.")

if __name__ == "__main__":
    main()
