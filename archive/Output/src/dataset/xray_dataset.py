"""
PyTorch Dataset class for X-Ray luggage scanner images.
Supports loading images with YOLO-format annotations and property vectors.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, List, Tuple


class XRayDataset(Dataset):
    """
    PyTorch Dataset for X-Ray security scanner images.

    Loads images with bounding box annotations (YOLO format) and
    optional property vector labels.

    Args:
        image_dir: Path to directory containing images.
        label_dir: Path to directory containing YOLO-format .txt labels.
        property_csv: Path to CSV with property vectors (from PropertyAnnotator).
        split_file: Path to .txt file listing image paths for this split.
        transform: Optional transform function (e.g., albumentations).
        image_size: Target image size (square).
        num_properties: Number of property dimensions (default 11).
        return_masks: Whether to generate bounding box masks.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str] = None,
        property_csv: Optional[str] = None,
        split_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: int = 640,
        num_properties: int = 11,
        return_masks: bool = False,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        self.num_properties = num_properties
        self.return_masks = return_masks

        # Load image file list
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                self.image_paths = [line.strip() for line in f if line.strip()]
        else:
            self.image_paths = sorted([
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

        # Load property annotations if available
        self.property_data = None
        if property_csv and os.path.exists(property_csv):
            self.property_data = pd.read_csv(property_csv)
            print(f"  ✓ Loaded {len(self.property_data)} property annotations")

        print(f"  ✓ XRayDataset initialized with {len(self.image_paths)} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
            - 'image': Tensor (C, H, W)
            - 'boxes': Tensor (N, 4) in xyxy format
            - 'labels': Tensor (N,) class IDs
            - 'properties': Tensor (N, num_properties) if property data available
            - 'image_path': str
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)

        if image is None:
            # Return a blank sample if image can't be loaded
            return self._blank_sample(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Load YOLO-format labels
        boxes, labels = self._load_labels(img_path, orig_h, orig_w)

        # Load property vectors
        properties = self._load_properties(img_path, len(boxes))

        # Apply augmentation/transform
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes if len(boxes) > 0 else [],
                class_labels=labels if len(labels) > 0 else [],
            )
            image = transformed["image"]
            if len(boxes) > 0:
                boxes = np.array(transformed["bboxes"])
                labels = np.array(transformed["class_labels"])

        # Resize to target size (letterbox)
        image, boxes = self._letterbox_resize(image, boxes, self.image_size)

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
                image = image.permute(2, 0, 1)  # HWC -> CHW
            image = image / 255.0  # Normalize to [0, 1]

        result = {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.zeros(0, dtype=torch.long),
            "properties": torch.tensor(properties, dtype=torch.float32),
            "image_path": img_path,
        }

        if self.return_masks:
            result["masks"] = self._generate_masks(boxes, self.image_size)

        return result

    def _load_labels(
        self, img_path: str, img_h: int, img_w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load YOLO-format bounding box labels."""
        # Determine label file path
        img_basename = os.path.splitext(os.path.basename(img_path))[0]

        label_path = None
        if self.label_dir:
            label_path = os.path.join(self.label_dir, f"{img_basename}.txt")
        else:
            # Try same directory as image
            label_path = os.path.join(os.path.dirname(img_path), f"{img_basename}.txt")

        boxes = []
        labels = []

        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]

                        # Convert from YOLO normalized to pixel xyxy
                        x1 = (cx - bw / 2) * img_w
                        y1 = (cy - bh / 2) * img_h
                        x2 = (cx + bw / 2) * img_w
                        y2 = (cy + bh / 2) * img_h

                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)

        return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)

    def _load_properties(
        self, img_path: str, num_objects: int
    ) -> np.ndarray:
        """Load property vectors for objects in this image."""
        if self.property_data is None or num_objects == 0:
            return np.zeros((max(num_objects, 0), self.num_properties), dtype=np.float32)

        # Match by image path
        img_props = self.property_data[
            self.property_data["image_path"] == img_path
        ]

        if len(img_props) == 0:
            return np.zeros((num_objects, self.num_properties), dtype=np.float32)

        # Extract property columns
        prop_columns = [
            "edge_sharpness", "length_to_width_ratio", "symmetry_score",
            "curvature_index", "approximate_volume", "material_category",
            "avg_absorption_intensity", "material_homogeneity",
            "density_level", "sharp_edge_count", "occlusion_score",
        ]

        available_cols = [c for c in prop_columns if c in img_props.columns]
        if available_cols:
            properties = img_props[available_cols].values.astype(np.float32)
        else:
            properties = np.zeros((len(img_props), self.num_properties), dtype=np.float32)

        # Pad or trim to match num_objects
        if len(properties) < num_objects:
            pad = np.zeros((num_objects - len(properties), properties.shape[1]), dtype=np.float32)
            properties = np.vstack([properties, pad])
        elif len(properties) > num_objects:
            properties = properties[:num_objects]

        return properties

    def _letterbox_resize(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image maintaining aspect ratio with padding."""
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h))

        # Pad to target size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top = pad_h // 2
        left = pad_w // 2

        image = cv2.copyMakeBorder(
            image, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # Adjust bounding boxes
        if len(boxes) > 0:
            boxes = boxes * scale
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top

        return image, boxes

    def _generate_masks(
        self, boxes: np.ndarray, img_size: int
    ) -> torch.Tensor:
        """Generate binary masks from bounding boxes."""
        num_objects = len(boxes)
        masks = torch.zeros((max(num_objects, 0), img_size, img_size))

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(c) for c in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_size, x2), min(img_size, y2)
            masks[i, y1:y2, x1:x2] = 1.0

        return masks

    def _blank_sample(self, img_path: str) -> Dict[str, torch.Tensor]:
        """Return a blank sample for failed image loads."""
        return {
            "image": torch.zeros((3, self.image_size, self.image_size)),
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros(0, dtype=torch.long),
            "properties": torch.zeros((0, self.num_properties)),
            "image_path": img_path,
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for DataLoader.

    Since different images may have different numbers of detections,
    boxes, labels, and properties are returned as lists, not stacked tensors.
    """
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "properties": [item["properties"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }
