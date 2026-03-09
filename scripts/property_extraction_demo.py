"""
Property Extraction Demo Script.

Takes an X-ray image, runs Model 1, and outputs annotated bounding boxes
with printed property vectors.

Usage:
    python scripts/property_extraction_demo.py \
        --image path/to/xray.jpg \
        --checkpoint checkpoints/stage2_best.pth \
        --output results/demo_output.jpg
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.pipeline import preprocess_xray, PreprocessingPipeline
from src.model1.architecture import PropertyYOLO
from src.dataset.property_annotator import PROPERTY_NAMES, MATERIAL_NAMES
from src.utils.visualization import visualize_detections


def run_demo(
    image_path: str,
    checkpoint_path: str = None,
    output_path: str = "demo_output.jpg",
    device: str = "auto",
    conf_threshold: float = 0.25,
):
    """
    Run Model 1 on a single X-ray image and visualize results.

    If no checkpoint is provided, runs preprocessing and property annotation
    without the neural network (for testing the pipeline).
    """
    print(f"\n{'='*60}")
    print(f"  X-Ray Property Extraction Demo")
    print(f"{'='*60}")

    # Load image
    print(f"\n📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    print(f"   Image size: {w}×{h}")

    # Preprocess
    print(f"\n🔧 Preprocessing (4-channel tensor construction)...")
    pipeline = PreprocessingPipeline(target_size=640)
    preprocessed = pipeline(image)
    print(f"   Output shape: {preprocessed.shape}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Run neural network inference
        print(f"\n🧠 Loading model from: {checkpoint_path}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = PropertyYOLO.load_checkpoint(checkpoint_path, device=device)
        model.eval()

        # Prepare input tensor
        tensor = torch.from_numpy(preprocessed).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
        tensor = tensor.to(device)

        with torch.no_grad():
            # Detection (using YOLOv8)
            # For demo, use the 3-channel version for detection
            det_input = tensor[:, :3, :, :]  # Use first 3 channels
            det_results = model.yolo.predict(
                det_input, conf=conf_threshold, verbose=False
            )

            # Property extraction
            prop_results = model.forward_properties(tensor)

        # Process results
        boxes = []
        labels = []
        scores = []
        property_vectors = []

        if det_results and len(det_results[0].boxes) > 0:
            det = det_results[0]
            for i, box in enumerate(det.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append(xyxy.tolist())
                labels.append(f"Class {int(box.cls)}")
                scores.append(float(box.conf))

            # Get property vectors
            if "properties" in prop_results:
                props = prop_results["properties"].cpu().numpy()
                for i in range(min(len(boxes), len(props))):
                    prop_dict = {}
                    for j, name in enumerate(PROPERTY_NAMES[:10]):
                        prop_dict[name] = float(props[i, j])
                    property_vectors.append(prop_dict)

        print(f"\n📊 Results: {len(boxes)} objects detected")
    else:
        # Demo without model — show preprocessing pipeline results
        print(f"\n⚠ No model checkpoint provided. Running pipeline demo only.")

        # Use simple thresholding for demo detection
        from src.dataset.property_annotator import PropertyAnnotator

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        annotator = PropertyAnnotator()

        # Simple blob detection for demo
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        labels = []
        scores = []
        property_vectors = []

        for cnt in contours[:5]:  # Top 5 largest regions
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip tiny regions
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox = [x, y, x + bw, y + bh]
            boxes.append(bbox)
            labels.append("unknown")
            scores.append(1.0)

            # Compute properties
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            props = annotator.compute_properties(gray, mask, bbox)
            property_vectors.append(props)

        print(f"\n📊 Results: {len(boxes)} regions detected (threshold-based)")

    # Visualize
    if len(boxes) > 0:
        fig = visualize_detections(
            image_rgb, boxes, labels,
            property_vectors=property_vectors if property_vectors else None,
            scores=scores,
            save_path=output_path,
        )
        print(f"\n💾 Output saved: {output_path}")

        # Print property vectors
        print(f"\n📋 Property Vectors:")
        for i, props in enumerate(property_vectors):
            print(f"\n   Object {i+1} ({labels[i]}):")
            for name, value in props.items():
                if name == "material_category":
                    mat_name = MATERIAL_NAMES[int(value)] if 0 <= int(value) < len(MATERIAL_NAMES) else "unknown"
                    print(f"     {name:30s} = {int(value)} ({mat_name})")
                elif name == "sharp_edge_count":
                    print(f"     {name:30s} = {int(value)}")
                else:
                    print(f"     {name:30s} = {value:.4f}")
    else:
        print(f"\n⚠ No objects detected.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-Ray Property Extraction Demo")
    parser.add_argument("--image", "-i", required=True, help="Path to input X-ray image")
    parser.add_argument("--checkpoint", "-c", default=None, help="Model checkpoint path")
    parser.add_argument("--output", "-o", default="demo_output.jpg", help="Output image path")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()
    run_demo(args.image, args.checkpoint, args.output, args.device, args.conf)
