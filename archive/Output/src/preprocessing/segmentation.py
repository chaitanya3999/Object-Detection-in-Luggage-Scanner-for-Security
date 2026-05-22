"""
Three-stage segmentation refinement for X-ray images.

Stage 1: Saliency detection (simplified — Otsu + morphological operations)
Stage 2: Mask R-CNN coarse masks (via detectron2, optional)
Stage 3: GrabCut refinement (cv2.grabCut)

Note: Full detectron2/U2-Net requires GPU and specific installations.
      This module provides fallback methods for CPU-only environments.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


class SegmentationRefinement:
    """
    Three-stage segmentation refinement pipeline.

    For production use, stages 1-2 should use U2-Net and Mask R-CNN respectively.
    This implementation provides OpenCV-based fallbacks for environments without
    detectron2 installed (e.g., local development without GPU).

    Usage:
        segmentor = SegmentationRefinement()
        masks = segmentor.segment(image, bboxes)
    """

    def __init__(
        self,
        use_detectron2: bool = False,
        detectron2_config: Optional[str] = None,
        detectron2_weights: Optional[str] = None,
        grabcut_iterations: int = 5,
    ):
        self.use_detectron2 = use_detectron2
        self.grabcut_iterations = grabcut_iterations
        self.detector = None

        if use_detectron2:
            self._init_detectron2(detectron2_config, detectron2_weights)

    def _init_detectron2(self, config_path: Optional[str], weights_path: Optional[str]):
        """Initialize Mask R-CNN with detectron2."""
        try:
            import detectron2
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo

            cfg = get_cfg()
            if config_path:
                cfg.merge_from_file(config_path)
            else:
                cfg.merge_from_file(model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                ))
            if weights_path:
                cfg.MODEL.WEIGHTS = weights_path
            else:
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cuda" if self._has_gpu() else "cpu"

            self.detector = DefaultPredictor(cfg)
            print("✓ Detectron2 Mask R-CNN initialized")
        except ImportError:
            print("⚠ detectron2 not installed, using OpenCV fallback")
            self.use_detectron2 = False

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def segment(
        self,
        image: np.ndarray,
        bboxes: List[List[int]],
    ) -> List[np.ndarray]:
        """
        Run full three-stage segmentation pipeline per bounding box.

        Args:
            image: Input image (H, W) grayscale or (H, W, 3) color.
            bboxes: List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            List of binary masks (H, W) for each bbox.
        """
        masks = []

        for bbox in bboxes:
            # Stage 1: Saliency-based initial segmentation
            saliency_mask = self._saliency_segmentation(image, bbox)

            # Stage 2: Mask R-CNN (if available) or use saliency result
            if self.use_detectron2 and self.detector is not None:
                rcnn_mask = self._maskrcnn_segmentation(image, bbox)
                # Combine: use RCNN if available, else saliency
                coarse_mask = rcnn_mask if rcnn_mask is not None else saliency_mask
            else:
                coarse_mask = saliency_mask

            # Stage 3: GrabCut refinement
            refined_mask = self._grabcut_refinement(image, bbox, coarse_mask)

            masks.append(refined_mask)

        return masks

    def _saliency_segmentation(
        self, image: np.ndarray, bbox: List[int]
    ) -> np.ndarray:
        """
        Stage 1: Saliency-based foreground detection.
        Uses Otsu thresholding + morphological operations as U2-Net fallback.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return mask

        # Extract ROI
        if len(image.shape) == 3:
            roi = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        else:
            roi = image[y1:y2, x1:x2]

        # Otsu thresholding for foreground
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        mask[y1:y2, x1:x2] = binary

        return mask

    def _maskrcnn_segmentation(
        self, image: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        """
        Stage 2: Mask R-CNN instance segmentation via detectron2.
        Returns None if no detection found for this bbox.
        """
        if self.detector is None:
            return None

        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image

        outputs = self.detector(image_color)
        instances = outputs["instances"]

        if len(instances) == 0:
            return None

        # Find best matching instance for this bbox
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
        pred_masks = instances.pred_masks.cpu().numpy()

        x1, y1, x2, y2 = bbox
        best_iou = 0
        best_mask = None

        for i in range(len(pred_boxes)):
            iou = self._compute_iou(bbox, pred_boxes[i].tolist())
            if iou > best_iou:
                best_iou = iou
                best_mask = pred_masks[i].astype(np.uint8) * 255

        return best_mask

    def _grabcut_refinement(
        self,
        image: np.ndarray,
        bbox: List[int],
        initial_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Stage 3: GrabCut refinement using initial mask.
        Refines the segmentation boundary using graph-cut optimization.
        """
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()

        h, w = image_color.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return initial_mask

        # Convert initial mask to GrabCut format
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        gc_mask[:] = cv2.GC_BGD  # Background

        # Set bbox region as probable foreground
        gc_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

        # Use initial mask to set definite foreground
        if initial_mask is not None:
            fg_pixels = initial_mask > 127
            gc_mask[fg_pixels] = cv2.GC_FGD

        # GrabCut models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        rect = (x1, y1, x2 - x1, y2 - y1)

        try:
            cv2.grabCut(
                image_color,
                gc_mask,
                rect,
                bgd_model,
                fgd_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_MASK if initial_mask is not None else cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error:
            # GrabCut can fail on very small regions
            return initial_mask

        # Extract foreground
        result = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        return result

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
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


def segment_objects(
    image: np.ndarray,
    bboxes: List[List[int]],
    use_detectron2: bool = False,
) -> List[np.ndarray]:
    """
    Convenience function: segment objects from bounding boxes.

    Args:
        image: Input image.
        bboxes: List of [x1, y1, x2, y2] bounding boxes.
        use_detectron2: Whether to try using detectron2.

    Returns:
        List of binary masks.
    """
    segmentor = SegmentationRefinement(use_detectron2=use_detectron2)
    return segmentor.segment(image, bboxes)
