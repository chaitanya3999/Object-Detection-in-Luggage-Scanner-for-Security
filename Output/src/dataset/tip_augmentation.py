"""
Threat Image Projection (TIP) augmentation for X-Ray images.

Uses Beer-Lambert law approximation to synthetically paste threat objects
into clean baggage images with physically-plausible blending.
"""

import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Optional
from pathlib import Path


class TIPAugmenter:
    """
    Threat Image Projection augmentation.

    Generates synthetic threat-positive X-ray images by compositing
    extracted threat objects onto clean baggage backgrounds using
    Beer-Lambert attenuation modeling.

    Usage:
        augmenter = TIPAugmenter(threat_bank_dir="data/threat_bank")
        augmented_img, augmented_labels = augmenter.augment(
            background_img, existing_labels
        )
    """

    def __init__(
        self,
        threat_bank_dir: Optional[str] = None,
        num_threats_range: Tuple[int, int] = (1, 3),
        rotation_range: Tuple[float, float] = (-180, 180),
        scale_range: Tuple[float, float] = (0.5, 1.5),
        opacity_range: Tuple[float, float] = (0.3, 0.8),
        seed: Optional[int] = None,
    ):
        """
        Args:
            threat_bank_dir: Directory containing extracted threat object images and masks.
            num_threats_range: Min/max number of threats to paste per image.
            rotation_range: Min/max rotation angle in degrees.
            scale_range: Min/max scale factor.
            opacity_range: Min/max opacity for Beer-Lambert blending.
            seed: Random seed for reproducibility.
        """
        self.threat_bank_dir = threat_bank_dir
        self.num_threats_range = num_threats_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.opacity_range = opacity_range
        self.rng = np.random.RandomState(seed)
        self.threat_bank = []

        if threat_bank_dir and os.path.exists(threat_bank_dir):
            self._load_threat_bank()

    def _load_threat_bank(self):
        """Load threat object images and masks from bank directory."""
        bank_dir = Path(self.threat_bank_dir)
        for img_path in sorted(bank_dir.glob("*.png")):
            mask_path = bank_dir / f"{img_path.stem}_mask.png"
            meta_path = bank_dir / f"{img_path.stem}_meta.json"

            threat_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if threat_img is None:
                continue

            mask = None
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            meta = {"class_id": 0, "class_name": "unknown"}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            self.threat_bank.append({
                "image": threat_img,
                "mask": mask,
                "class_id": meta.get("class_id", 0),
                "class_name": meta.get("class_name", "unknown"),
            })

        print(f"✓ Loaded {len(self.threat_bank)} threat objects from bank")

    def extract_threat_from_annotation(
        self,
        image: np.ndarray,
        bbox: List[int],
        class_id: int,
        class_name: str = "unknown",
        output_dir: Optional[str] = None,
        idx: int = 0,
    ) -> dict:
        """
        Extract a threat object from an annotated image to add to the bank.

        Args:
            image: Source X-ray image (grayscale).
            bbox: [x1, y1, x2, y2] bounding box.
            class_id: Class label ID.
            class_name: Class name string.
            output_dir: Optional directory to save extracted threat.
            idx: Index for naming.

        Returns:
            Threat bank entry dict.
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        cropped = image[y1:y2, x1:x2].copy()

        # Create approximate mask using Otsu thresholding
        _, mask = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        entry = {
            "image": cropped,
            "mask": mask,
            "class_id": class_id,
            "class_name": class_name,
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            name = f"threat_{class_name}_{idx:04d}"
            cv2.imwrite(os.path.join(output_dir, f"{name}.png"), cropped)
            cv2.imwrite(os.path.join(output_dir, f"{name}_mask.png"), mask)
            with open(os.path.join(output_dir, f"{name}_meta.json"), "w") as f:
                json.dump({"class_id": class_id, "class_name": class_name}, f)

        self.threat_bank.append(entry)
        return entry

    def augment(
        self,
        background: np.ndarray,
        existing_labels: Optional[List[dict]] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Paste threat objects onto a background image using Beer-Lambert blending.

        Args:
            background: Background X-ray image (grayscale, H×W).
            existing_labels: Existing labels in the image (list of dicts with
                           'bbox' and 'class_id' keys).

        Returns:
            Tuple of (augmented_image, all_labels) where labels include both
            existing and newly pasted threat labels.
        """
        if len(self.threat_bank) == 0:
            return background.copy(), existing_labels or []

        result = background.copy().astype(np.float32)
        h, w = result.shape[:2]
        labels = list(existing_labels or [])

        num_threats = self.rng.randint(
            self.num_threats_range[0], self.num_threats_range[1] + 1
        )

        for _ in range(num_threats):
            # Pick a random threat
            threat = self.rng.choice(self.threat_bank)
            threat_img = threat["image"].astype(np.float32)
            threat_mask = threat["mask"]

            # Random rotation
            angle = self.rng.uniform(*self.rotation_range)
            threat_img, threat_mask = self._rotate(threat_img, threat_mask, angle)

            # Random scale
            scale = self.rng.uniform(*self.scale_range)
            new_h = max(int(threat_img.shape[0] * scale), 1)
            new_w = max(int(threat_img.shape[1] * scale), 1)
            threat_img = cv2.resize(threat_img, (new_w, new_h))
            if threat_mask is not None:
                threat_mask = cv2.resize(threat_mask, (new_w, new_h))

            # Ensure threat fits in the background
            if new_h >= h or new_w >= w:
                continue

            # Random position
            y_pos = self.rng.randint(0, h - new_h)
            x_pos = self.rng.randint(0, w - new_w)

            # Beer-Lambert blending
            opacity = self.rng.uniform(*self.opacity_range)
            result = self._beer_lambert_blend(
                result, threat_img, threat_mask, x_pos, y_pos, opacity
            )

            # Add label
            labels.append({
                "class_id": threat["class_id"],
                "class_name": threat["class_name"],
                "bbox": [x_pos, y_pos, x_pos + new_w, y_pos + new_h],
            })

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result, labels

    def _rotate(
        self, image: np.ndarray, mask: Optional[np.ndarray], angle: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Rotate image and mask by angle degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding box
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(image, M, (new_w, new_h), borderValue=0)
        rotated_mask = None
        if mask is not None:
            rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), borderValue=0)

        return rotated_img, rotated_mask

    def _beer_lambert_blend(
        self,
        background: np.ndarray,
        threat: np.ndarray,
        mask: Optional[np.ndarray],
        x: int,
        y: int,
        opacity: float,
    ) -> np.ndarray:
        """
        Blend threat into background using Beer-Lambert attenuation model.

        In X-ray imaging, pixel intensity I = I0 * exp(-μ*t) where μ is the
        attenuation coefficient and t is the material thickness. Overlapping
        objects multiply their transmission factors.

        For grayscale X-ray: combined = background * threat / 255 * opacity_factor
        """
        th, tw = threat.shape[:2]
        roi = background[y : y + th, x : x + tw]

        if mask is not None:
            alpha = mask.astype(np.float32) / 255.0
        else:
            alpha = np.ones_like(threat, dtype=np.float32)

        # Beer-Lambert: multiplicative blending
        # Normalize threat to transmission coefficient
        transmission = threat / 255.0

        # Combined transmission = background_transmission * threat_transmission
        blended = roi * (1.0 - alpha * opacity) + roi * transmission * alpha * opacity

        background[y : y + th, x : x + tw] = blended
        return background

    def generate_batch(
        self,
        backgrounds: List[np.ndarray],
        num_augmented_per_bg: int = 10,
    ) -> List[Tuple[np.ndarray, List[dict]]]:
        """
        Generate a batch of augmented images.

        Args:
            backgrounds: List of clean background X-ray images.
            num_augmented_per_bg: How many augmented versions per background.

        Returns:
            List of (augmented_image, labels) tuples.
        """
        results = []
        for bg in backgrounds:
            for _ in range(num_augmented_per_bg):
                aug_img, aug_labels = self.augment(bg)
                results.append((aug_img, aug_labels))
        return results


def labels_to_yolo(labels: List[dict], img_h: int, img_w: int) -> List[str]:
    """
    Convert labels to YOLO format strings.

    Args:
        labels: List of dicts with 'class_id' and 'bbox' [x1,y1,x2,y2].
        img_h, img_w: Image dimensions.

    Returns:
        List of YOLO format strings: "class_id cx cy w h" (normalized).
    """
    lines = []
    for label in labels:
        x1, y1, x2, y2 = label["bbox"]
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{label['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines
