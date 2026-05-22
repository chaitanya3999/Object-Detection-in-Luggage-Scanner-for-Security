"""
Multi-task loss function for Model 1.

Combines:
  - Detection loss (from YOLOv8)
  - Property regression loss (MSE)
  - Material classification loss (CrossEntropy)

Uses Uncertainty Weighting (Kendall et al., 2018) to automatically
balance loss contributions across tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class UncertaintyWeight(nn.Module):
    """
    Learnable uncertainty parameter for a task.

    Implements Kendall et al., 2018: "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics"

    The loss is scaled as: L_task / (2 * sigma^2) + log(sigma)
    where sigma is a learnable parameter. This automatically
    down-weights noisy tasks and up-weights reliable ones.
    """

    def __init__(self, init_sigma: float = 1.0):
        super().__init__()
        # We learn log(sigma^2) for numerical stability
        self.log_sigma_sq = nn.Parameter(
            torch.tensor(2.0 * torch.tensor(init_sigma).log())
        )

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by uncertainty weight."""
        precision = torch.exp(-self.log_sigma_sq)
        weighted_loss = precision * loss + self.log_sigma_sq
        return weighted_loss

    @property
    def sigma(self) -> float:
        """Current sigma value."""
        return (0.5 * self.log_sigma_sq).exp().item()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p) = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for the dual-head YOLOv8 model.

    Combines detection, property regression, and material classification
    losses with automatic uncertainty weighting.

    Usage:
        criterion = MultiTaskLoss(
            num_properties=10,
            num_materials=4,
            use_uncertainty_weighting=True,
        )
        loss, loss_dict = criterion(predictions, targets)
    """

    def __init__(
        self,
        num_properties: int = 10,
        num_materials: int = 4,
        use_uncertainty_weighting: bool = True,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        property_loss_type: str = "mse",  # "mse" or "smooth_l1"
    ):
        super().__init__()

        self.num_properties = num_properties
        self.num_materials = num_materials
        self.use_uncertainty = use_uncertainty_weighting
        self.property_loss_type = property_loss_type

        # Property regression loss
        if property_loss_type == "mse":
            self.property_loss_fn = nn.MSELoss()
        elif property_loss_type == "smooth_l1":
            self.property_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown property loss: {property_loss_type}")

        # Material classification loss
        if use_focal_loss:
            self.material_loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.material_loss_fn = nn.CrossEntropyLoss()

        # Uncertainty weights
        if use_uncertainty_weighting:
            self.uw_detection = UncertaintyWeight(init_sigma=1.0)
            self.uw_property = UncertaintyWeight(init_sigma=1.0)
            self.uw_material = UncertaintyWeight(init_sigma=1.0)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        detection_loss: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Compute combined multi-task loss.

        Args:
            predictions: Dict with keys:
                - 'properties': (N, num_properties) predicted property vectors
                - 'material_logits': (N, num_materials) material class logits (optional)
            targets: Dict with keys:
                - 'properties': (N, num_properties) ground truth property vectors
                - 'material_labels': (N,) ground truth material class indices (optional)
            detection_loss: Optional detection loss from YOLOv8 (precomputed).

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))

        # Detection loss (from YOLOv8, passed in)
        if detection_loss is not None:
            if self.use_uncertainty:
                det_loss = self.uw_detection(detection_loss)
            else:
                det_loss = detection_loss
            total_loss = total_loss + det_loss
            loss_dict["detection"] = detection_loss.item()
            loss_dict["detection_weighted"] = det_loss.item()

        # Property regression loss
        if "properties" in predictions and "properties" in targets:
            pred_props = predictions["properties"]
            gt_props = targets["properties"]

            if pred_props.shape[0] > 0 and gt_props.shape[0] > 0:
                # Separate continuous and categorical properties
                # Continuous: indices 0-4, 6-8, 9  |  Categorical: index 5
                continuous_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9]
                cat_idx = 5

                prop_loss = self.property_loss_fn(
                    pred_props[:, continuous_idx],
                    gt_props[:, continuous_idx],
                )

                if self.use_uncertainty:
                    prop_loss = self.uw_property(prop_loss)

                total_loss = total_loss + prop_loss
                loss_dict["property_regression"] = prop_loss.item()

        # Material classification loss
        if (
            "material_logits" in predictions
            and "material_labels" in targets
        ):
            mat_logits = predictions["material_logits"]
            mat_labels = targets["material_labels"]

            if mat_logits.shape[0] > 0 and mat_labels.shape[0] > 0:
                mat_loss = self.material_loss_fn(mat_logits, mat_labels.long())

                if self.use_uncertainty:
                    mat_loss = self.uw_material(mat_loss)

                total_loss = total_loss + mat_loss
                loss_dict["material_classification"] = mat_loss.item()

        loss_dict["total"] = total_loss.item()

        # Log uncertainty weights
        if self.use_uncertainty:
            loss_dict["sigma_detection"] = self.uw_detection.sigma
            loss_dict["sigma_property"] = self.uw_property.sigma
            loss_dict["sigma_material"] = self.uw_material.sigma

        return total_loss, loss_dict

    def _get_device(self, predictions: Dict[str, torch.Tensor]) -> torch.device:
        """Get the device from predictions."""
        for v in predictions.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")
