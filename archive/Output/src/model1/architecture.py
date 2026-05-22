"""
Modified YOLOv8 architecture with Property Regression Head.

Extends Ultralytics YOLOv8 with:
  - Head 1: Standard detection head (bbox + objectness + class)
  - Head 2: Property Regression Head (3-layer MLP: 512→256→10)
  - Material Branch: 4-class material classifier (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class PropertyRegressionHead(nn.Module):
    """
    3-layer MLP that regresses the 10-dimensional continuous property vector
    from the same feature maps used by the detection head.

    Architecture: input_dim → 512 → BN → ReLU → 256 → BN → ReLU → 10
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [512, 256],
        num_properties: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, num_properties))

        self.mlp = nn.Sequential(*layers)

        # Sigmoid for bounded [0,1] properties, applied selectively
        self.num_properties = num_properties

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, input_dim) pooled ROI features, or
                      (B, input_dim, H, W) feature map (will be GAP'ed).

        Returns:
            (N, num_properties) predicted property vectors.
        """
        if features.dim() == 4:
            # Global Average Pooling
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)

        props = self.mlp(features)

        # Apply sigmoid to properties that should be in [0, 1]
        # Properties 0-4, 6-8: float [0,1], property 5 (material): categorical,
        # property 9 (sharp_edge_count): integer
        props_bounded = torch.sigmoid(props[:, :5])
        props_material = props[:, 5:6]  # Will use cross-entropy separately
        props_bounded2 = torch.sigmoid(props[:, 6:9])
        props_count = torch.relu(props[:, 9:10])  # Non-negative integer count

        return torch.cat([props_bounded, props_material, props_bounded2, props_count], dim=1)


class MaterialClassificationBranch(nn.Module):
    """
    Shallow CNN branch for 4-class material classification
    (organic / metallic / mixed / opaque).

    Operates on the material-discriminative channel (HE-LE difference).
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) material channel (HE-LE difference).

        Returns:
            (B, num_classes) logits.
        """
        features = self.conv(x).flatten(1)
        return self.fc(features)


class PropertyYOLO(nn.Module):
    """
    Dual-head YOLOv8: detection + property extraction.

    Wraps Ultralytics YOLOv8 model and adds a PropertyRegressionHead
    that branches from the backbone feature maps.

    Usage:
        model = PropertyYOLO(model_size="yolov8m", num_classes=6)
        # In training:
        det_loss = model.train_detection(images, targets)  # Stage 1
        prop_loss = model.train_properties(images, property_targets)  # Stage 2
        # In inference:
        detections, property_vectors = model(images)
    """

    def __init__(
        self,
        model_size: str = "yolov8m",
        num_classes: int = 6,
        num_properties: int = 10,
        input_channels: int = 4,
        pretrained: bool = True,
        property_head_dims: List[int] = [512, 256],
        property_dropout: float = 0.1,
        material_branch: bool = True,
        num_materials: int = 4,
    ):
        super().__init__()

        self.num_properties = num_properties
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.material_branch_enabled = material_branch

        # Initialize YOLOv8 backbone
        self._init_yolo(model_size, num_classes, pretrained, input_channels)

        # Determine backbone feature dimension
        self.backbone_dim = self._get_backbone_dim()

        # Property Regression Head
        self.property_head = PropertyRegressionHead(
            input_dim=self.backbone_dim,
            hidden_dims=property_head_dims,
            num_properties=num_properties,
            dropout=property_dropout,
        )

        # Material Classification Branch
        self.material_branch = None
        if material_branch:
            self.material_branch = MaterialClassificationBranch(
                input_channels=1,
                num_classes=num_materials,
            )

        # Stage tracking
        self.training_stage = 1  # 1 = detection, 2 = property regression

    def _init_yolo(self, model_size: str, num_classes: int, pretrained: bool, input_channels: int):
        """Initialize the Ultralytics YOLOv8 model."""
        from ultralytics import YOLO

        # Load pretrained model
        if pretrained:
            self.yolo = YOLO(f"{model_size}.pt")
        else:
            self.yolo = YOLO(f"{model_size}.yaml")

        # Modify first conv layer for 4-channel input if needed
        if input_channels != 3:
            self._modify_input_channels(input_channels)

    def _modify_input_channels(self, new_channels: int):
        """
        Modify the first convolutional layer to accept a different number
        of input channels while preserving pretrained weights where possible.
        """
        model = self.yolo.model

        # Access the first conv layer
        first_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break

        if first_conv is not None and first_conv.in_channels != new_channels:
            old_weight = first_conv.weight.data
            new_conv = nn.Conv2d(
                new_channels,
                first_conv.out_channels,
                first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            # Initialize new weights
            with torch.no_grad():
                if new_channels > first_conv.in_channels:
                    # Copy existing weights and initialize extra channels
                    new_conv.weight[:, :first_conv.in_channels] = old_weight
                    nn.init.kaiming_normal_(
                        new_conv.weight[:, first_conv.in_channels:],
                        mode="fan_out",
                    )
                else:
                    new_conv.weight = nn.Parameter(old_weight[:, :new_channels])

                if first_conv.bias is not None:
                    new_conv.bias = nn.Parameter(first_conv.bias.data.clone())

            # Replace the conv layer
            for name, module in model.named_modules():
                if module is first_conv:
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = dict(model.named_modules())[parent_name] if parent_name else model
                    setattr(parent, child_name, new_conv)
                    break

    def _get_backbone_dim(self) -> int:
        """Determine the backbone output feature dimension."""
        # YOLOv8 sizes and their channel widths at the neck output
        # This is approximate — exact value depends on model size
        try:
            # Try to get from model
            model = self.yolo.model
            for module in reversed(list(model.modules())):
                if isinstance(module, nn.Conv2d):
                    return module.out_channels
        except Exception:
            pass
        return 512  # Default fallback

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps from YOLOv8 backbone.

        Args:
            images: (B, C, H, W) input tensor.

        Returns:
            (B, backbone_dim, H', W') feature maps.
        """
        model = self.yolo.model

        # Run through backbone layers to get feature maps
        x = images
        features = None

        for i, layer in enumerate(model.model):
            x = layer(x)
            # Store intermediate features
            if hasattr(layer, 'f') and layer.f == -1:
                features = x

        # Return the last backbone feature map
        if features is None:
            features = x

        return features

    def forward_properties(
        self,
        images: torch.Tensor,
        roi_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for property extraction.

        Args:
            images: (B, C, H, W) preprocessed X-ray images.
            roi_features: Optional pre-extracted ROI features.

        Returns:
            Dict with 'properties' and optionally 'material_logits'.
        """
        if roi_features is None:
            roi_features = self.extract_features(images)

        result = {}

        # Property regression
        result["properties"] = self.property_head(roi_features)

        # Material classification
        if self.material_branch is not None and images.shape[1] >= 4:
            # Use the 4th channel (HE-LE) for material prediction
            material_input = images[:, 3:4, :, :]
            result["material_logits"] = self.material_branch(material_input)

        return result

    def set_training_stage(self, stage: int):
        """
        Set the training stage.

        Stage 1: Train detection only (freeze property head)
        Stage 2: Train property head only (freeze backbone)
        """
        self.training_stage = stage

        if stage == 1:
            # Freeze property head, train detection backbone
            for param in self.property_head.parameters():
                param.requires_grad = False
            if self.material_branch:
                for param in self.material_branch.parameters():
                    param.requires_grad = False
            print("⚙ Stage 1: Training detection backbone only")

        elif stage == 2:
            # Freeze backbone, train property head
            for param in self.yolo.model.parameters():
                param.requires_grad = False
            for param in self.property_head.parameters():
                param.requires_grad = True
            if self.material_branch:
                for param in self.material_branch.parameters():
                    param.requires_grad = True
            print("⚙ Stage 2: Training property head only (backbone frozen)")

    def get_trainable_params(self) -> List[torch.Tensor]:
        """Get parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def save_checkpoint(self, path: str, epoch: int, optimizer=None, metrics=None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "num_properties": self.num_properties,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "training_stage": self.training_stage,
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "PropertyYOLO":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_properties=checkpoint.get("num_properties", 10),
            num_classes=checkpoint.get("num_classes", 6),
            input_channels=checkpoint.get("input_channels", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.training_stage = checkpoint.get("training_stage", 1)
        return model
