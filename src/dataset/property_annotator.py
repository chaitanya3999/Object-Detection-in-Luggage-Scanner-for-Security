"""
Semi-automated Property Annotator for X-Ray images.

Computes the 11-dimensional property vector for each detected object
using OpenCV and NumPy operations on the object's mask and bounding box.

Property Schema (10 + 1 occlusion):
  0. Edge Sharpness Score        (float, 0-1)
  1. Length-to-Width Ratio       (float, 0-10+)
  2. Symmetry Score              (float, 0-1)
  3. Curvature Index             (float, 0-1)
  4. Approximate Volume          (float, 0-1, normalized)
  5. Material Category           (int, 0-3: organic/metallic/mixed/opaque)
  6. Avg Absorption Intensity    (float, 0-1)
  7. Material Homogeneity        (float, 0-1)
  8. Density Level               (float, 0-1)
  9. Sharp Edge Count            (int, 0-20)
 10. Occlusion Score             (float, 0-1)
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


# Material category mapping
MATERIAL_ORGANIC = 0
MATERIAL_METALLIC = 1
MATERIAL_MIXED = 2
MATERIAL_OPAQUE = 3

MATERIAL_NAMES = ["organic", "metallic", "mixed", "opaque"]

PROPERTY_NAMES = [
    "edge_sharpness",
    "length_to_width_ratio",
    "symmetry_score",
    "curvature_index",
    "approximate_volume",
    "material_category",
    "avg_absorption_intensity",
    "material_homogeneity",
    "density_level",
    "sharp_edge_count",
    "occlusion_score",
]


class PropertyAnnotator:
    """
    Computes 11-dimensional property vectors from X-ray images.

    Usage:
        annotator = PropertyAnnotator()
        img_gray = cv2.imread("xray.jpg", cv2.IMREAD_GRAYSCALE)
        mask = ...  # Binary mask of the object
        bbox = [x1, y1, x2, y2]
        properties = annotator.compute_properties(img_gray, mask, bbox)
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        corner_max_corners: int = 100,
        corner_quality: float = 0.01,
        corner_min_distance: int = 10,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.corner_max_corners = corner_max_corners
        self.corner_quality = corner_quality
        self.corner_min_distance = corner_min_distance

    def compute_properties(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[int],
        all_bboxes: Optional[List[List[int]]] = None,
        current_idx: int = 0,
    ) -> Dict[str, float]:
        """
        Compute the full 11-dimensional property vector for an object.

        Args:
            image: Grayscale X-ray image (H, W), uint8.
            mask: Binary mask for this object (H, W), uint8 (0 or 255).
            bbox: Bounding box [x1, y1, x2, y2].
            all_bboxes: All bounding boxes in the image (for occlusion computation).
            current_idx: Index of the current bbox in all_bboxes.

        Returns:
            Dictionary mapping property names to their values.
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))

        # Extract object region
        obj_img = image[y1:y2, x1:x2]
        obj_mask = mask[y1:y2, x1:x2] if mask is not None else np.ones_like(obj_img)

        # Ensure mask is binary
        if obj_mask.max() > 1:
            obj_mask = (obj_mask > 127).astype(np.uint8)
        else:
            obj_mask = obj_mask.astype(np.uint8)

        # Compute each property
        props = {}
        props["edge_sharpness"] = self._edge_sharpness(obj_img, obj_mask)
        props["length_to_width_ratio"] = self._length_to_width_ratio(x1, y1, x2, y2)
        props["symmetry_score"] = self._symmetry_score(obj_img, obj_mask)
        props["curvature_index"] = self._curvature_index(obj_mask)
        props["approximate_volume"] = self._approximate_volume(obj_img, obj_mask)
        props["material_category"] = self._material_category(obj_img, obj_mask)
        props["avg_absorption_intensity"] = self._avg_absorption(obj_img, obj_mask)
        props["material_homogeneity"] = self._material_homogeneity(obj_img, obj_mask)
        props["density_level"] = self._density_level(obj_img, obj_mask)
        props["sharp_edge_count"] = self._sharp_edge_count(obj_img, obj_mask)
        props["occlusion_score"] = self._occlusion_score(
            bbox, all_bboxes, current_idx
        )

        return props

    def compute_properties_vector(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[int],
        all_bboxes: Optional[List[List[int]]] = None,
        current_idx: int = 0,
    ) -> np.ndarray:
        """Compute property vector as a numpy array (11,)."""
        props = self.compute_properties(image, mask, bbox, all_bboxes, current_idx)
        return np.array([props[name] for name in PROPERTY_NAMES], dtype=np.float32)

    # ----------------------------------------------------------------
    # Individual property computation methods
    # ----------------------------------------------------------------

    def _edge_sharpness(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> float:
        """Canny edge detection density within the object mask."""
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return 0.0
        edges = cv2.Canny(obj_img, self.canny_low, self.canny_high)
        masked_edges = edges * obj_mask
        edge_pixels = np.count_nonzero(masked_edges)
        mask_pixels = np.count_nonzero(obj_mask)
        return float(np.clip(edge_pixels / max(mask_pixels, 1), 0.0, 1.0))

    def _length_to_width_ratio(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Bounding box length-to-width ratio."""
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        ratio = max(w, h) / min(w, h)
        return float(min(ratio, 10.0))

    def _symmetry_score(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> float:
        """Symmetry via horizontal pixel distribution comparison."""
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return 0.0

        masked = obj_img.astype(float) * obj_mask
        h, w = masked.shape

        # Horizontal symmetry
        if w < 2:
            return 1.0
        left = masked[:, : w // 2]
        right = np.flip(masked[:, (w - w // 2):], axis=1)

        # Make same size
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

        if left.size == 0:
            return 0.0

        diff = np.abs(left - right).mean()
        max_val = max(masked.max(), 1.0)
        symmetry = 1.0 - (diff / max_val)
        return float(np.clip(symmetry, 0.0, 1.0))

    def _curvature_index(self, obj_mask: np.ndarray) -> float:
        """Contour curvature analysis — ratio of contour length to convex hull length."""
        if obj_mask.size == 0 or obj_mask.sum() == 0:
            return 0.0

        mask_uint8 = (obj_mask * 255).astype(np.uint8) if obj_mask.max() <= 1 else obj_mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        hull = cv2.convexHull(largest)
        hull_perimeter = cv2.arcLength(hull, True)

        if hull_perimeter == 0:
            return 0.0

        # Curvature index: how much the contour deviates from its convex hull
        curvature = 1.0 - (hull_perimeter / max(perimeter, 1.0))
        return float(np.clip(curvature, 0.0, 1.0))

    def _approximate_volume(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> float:
        """Pixel area × opacity-estimated depth, normalized."""
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return 0.0

        masked = obj_img.astype(float) * obj_mask
        area = np.count_nonzero(obj_mask)
        avg_intensity = masked.sum() / max(area, 1)

        # Approximate depth from intensity (darker = denser = thicker)
        # Invert: higher intensity in X-ray = less absorption = thinner
        depth_estimate = 1.0 - (avg_intensity / 255.0)

        # Normalize volume: area * depth / max_possible
        max_area = obj_img.shape[0] * obj_img.shape[1]
        volume = (area * depth_estimate) / max(max_area, 1)
        return float(np.clip(volume, 0.0, 1.0))

    def _material_category(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> int:
        """
        Estimate material category from intensity distribution.

        For single-energy images, we use intensity thresholds as a proxy:
        - Very dark (high absorption): metallic (1)
        - Light (low absorption): organic (0)
        - Mixed range: mixed (2)
        - Uniform very dark: opaque (3)
        """
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return MATERIAL_MIXED

        masked_pixels = obj_img[obj_mask > 0].astype(float)
        if len(masked_pixels) == 0:
            return MATERIAL_MIXED

        mean_val = masked_pixels.mean() / 255.0
        std_val = masked_pixels.std() / 255.0

        if mean_val < 0.2 and std_val < 0.1:
            return MATERIAL_OPAQUE
        elif mean_val < 0.4:
            return MATERIAL_METALLIC
        elif mean_val > 0.6:
            return MATERIAL_ORGANIC
        else:
            return MATERIAL_MIXED

    def _avg_absorption(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> float:
        """Mean pixel value in grayscale channel, normalized to [0, 1]."""
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return 0.0
        masked_pixels = obj_img[obj_mask > 0].astype(float)
        if len(masked_pixels) == 0:
            return 0.0
        return float(masked_pixels.mean() / 255.0)

    def _material_homogeneity(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> float:
        """Standard deviation of absorption across object, inverted to [0, 1]."""
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return 0.0
        masked_pixels = obj_img[obj_mask > 0].astype(float)
        if len(masked_pixels) < 2:
            return 1.0
        std = masked_pixels.std() / 255.0
        # Invert: high homogeneity = low std = score close to 1
        return float(np.clip(1.0 - std * 2, 0.0, 1.0))

    def _density_level(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> float:
        """
        Density level from weighted intensity ratio.
        For single-energy, approximated as inverted mean intensity.
        Dark objects have higher density.
        """
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return 0.0
        masked_pixels = obj_img[obj_mask > 0].astype(float)
        if len(masked_pixels) == 0:
            return 0.0
        # Invert: darker = higher density
        return float(1.0 - masked_pixels.mean() / 255.0)

    def _sharp_edge_count(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> int:
        """Harris / Shi-Tomasi corner detection count."""
        if obj_img.size == 0 or obj_mask.sum() == 0:
            return 0

        # Shi-Tomasi corner detection
        masked_img = obj_img * obj_mask
        corners = cv2.goodFeaturesToTrack(
            masked_img,
            maxCorners=self.corner_max_corners,
            qualityLevel=self.corner_quality,
            minDistance=self.corner_min_distance,
        )

        count = 0 if corners is None else len(corners)
        return min(count, 20)

    def _occlusion_score(
        self,
        bbox: List[int],
        all_bboxes: Optional[List[List[int]]],
        current_idx: int,
    ) -> float:
        """Fraction of bounding box area overlapped by other bounding boxes."""
        if all_bboxes is None or len(all_bboxes) <= 1:
            return 0.0

        x1, y1, x2, y2 = bbox
        bbox_area = max((x2 - x1) * (y2 - y1), 1)
        total_overlap = 0.0

        for i, other_bbox in enumerate(all_bboxes):
            if i == current_idx:
                continue
            ox1, oy1, ox2, oy2 = other_bbox

            # Compute intersection
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                total_overlap += intersection

        return float(np.clip(total_overlap / bbox_area, 0.0, 1.0))


def batch_annotate(
    image_paths: List[str],
    annotation_paths: List[str],
    output_csv: str,
    annotator: Optional[PropertyAnnotator] = None,
):
    """
    Batch compute property vectors for a dataset.

    Args:
        image_paths: List of image file paths.
        annotation_paths: List of annotation files (YOLO format .txt).
        output_csv: Output CSV path for the property manifest.
        annotator: PropertyAnnotator instance (created if None).
    """
    import pandas as pd
    from tqdm import tqdm

    if annotator is None:
        annotator = PropertyAnnotator()

    records = []

    for img_path, ann_path in tqdm(
        zip(image_paths, annotation_paths),
        total=len(image_paths),
        desc="Computing properties",
    ):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠ Could not read: {img_path}")
            continue

        h, w = image.shape[:2]

        # Parse YOLO annotations
        bboxes = []
        classes = []
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        # Convert YOLO to pixel coords
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        bboxes.append([x1, y1, x2, y2])
                        classes.append(cls)

        # Create simple mask from bbox (when segmentation masks unavailable)
        for i, (bbox, cls) in enumerate(zip(bboxes, classes)):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            bx1, by1, bx2, by2 = bbox
            mask[max(0, by1):by2, max(0, bx1):bx2] = 1

            props = annotator.compute_properties(
                image, mask, bbox, all_bboxes=bboxes, current_idx=i
            )

            record = {
                "image_path": img_path,
                "class_id": cls,
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
            }
            record.update(props)
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(records)} property annotations to {output_csv}")
    return df
