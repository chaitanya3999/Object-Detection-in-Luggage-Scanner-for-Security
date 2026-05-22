"""
Modified YOLOv8 architecture with Property Regression Head (ROI-Aligned).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import Optional, Dict, List, Tuple

class PropertyRegressionHead(nn.Module):
    def __init__(self, in_channels: int = 512, output_dim: int = 11):
        super().__init__()
        self.roi_size = 7
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * self.roi_size * self.roi_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )
        self.num_properties = output_dim

    def forward(self, feature_map: torch.Tensor, boxes: List[torch.Tensor], img_size: Tuple[int, int]) -> torch.Tensor:
        rois = []
        for i, b in enumerate(boxes):
            if len(b) > 0:
                idx_col = torch.full((len(b), 1), i, dtype=b.dtype, device=b.device)
                rois.append(torch.cat([idx_col, b], dim=1))
        
        if not rois:
            return torch.zeros(0, self.num_properties, device=feature_map.device)
            
        rois = torch.cat(rois, dim=0)
        spatial_scale = feature_map.shape[-1] / img_size[1] 
        
        pooled = roi_align(feature_map, rois, self.roi_size, spatial_scale)
        flat = pooled.flatten(1)
        props = self.mlp(flat)

        props_bounded = torch.sigmoid(props[:, :5])
        props_material = props[:, 5:6]  
        props_bounded2 = torch.sigmoid(props[:, 6:9])
        props_count = torch.relu(props[:, 9:10])  
        props_11 = torch.sigmoid(props[:, 10:11])

        return torch.cat([props_bounded, props_material, props_bounded2, props_count, props_11], dim=1)

class MaterialClassificationBranch(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))

class PropertyYOLO(nn.Module):
    def __init__(self, model_size="yolov8m", num_classes=6, num_properties=11, input_channels=4, pretrained=True, weights_path=None):
        super().__init__()
        self.num_properties, self.num_classes, self.input_channels = num_properties, num_classes, input_channels
        
        # Pass weights_path down to the YOLO initializer
        self._init_yolo(model_size, num_classes, pretrained, input_channels, weights_path)
        
        self.property_head = PropertyRegressionHead(in_channels=192, output_dim=num_properties)
        self.material_branch = MaterialClassificationBranch(input_channels=1, num_classes=4)
        self.training_stage = 1

    def train(self, mode: bool = True):
        self.training = mode
        self.property_head.train(mode)
        if self.material_branch: self.material_branch.train(mode)
        if hasattr(self, 'yolo'):
            self.yolo.model.eval()
            for param in self.yolo.model.parameters(): param.requires_grad = False
        return self

    def eval(self):
        self.training = False
        self.property_head.eval()
        if self.material_branch: self.material_branch.eval()
        if hasattr(self, 'yolo'): self.yolo.model.eval()
        return self

    def _init_yolo(self, model_size, num_classes, pretrained, input_channels, weights_path):
        from ultralytics import YOLO
        import ultralytics, yaml
        from pathlib import Path
        
        yaml_path = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8.yaml"
        custom_yaml_path = f"custom_4ch_{model_size}.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f: d = yaml.safe_load(f)
            d["ch"], d["nc"] = input_channels, num_classes
            with open(custom_yaml_path, "w") as f: yaml.dump(d, f)
            self.yolo = YOLO(custom_yaml_path)
            
        if pretrained and weights_path:
            self.yolo.load(weights_path)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        y, x = [], images
        for m in self.yolo.model.model:
            if m.__class__.__name__ == 'Detect': break
            if m.f != -1: x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return y[15] # Semantically rich PANet neck layer

    def forward_properties(self, images: torch.Tensor, boxes: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        roi_features = self.extract_features(images)
        result = {"properties": self.property_head(roi_features, boxes, (images.shape[2], images.shape[3]))}
        if self.material_branch is not None and images.shape[1] >= 4:
            result["material_logits"] = self.material_branch(images[:, 3:4, :, :])
        return result

    def get_trainable_params(self): return [p for p in self.parameters() if p.requires_grad]
    def set_training_stage(self, stage): pass # Handled by train() override now

    def save_checkpoint(self, path: str, epoch: int, optimizer=None, metrics=None):
        checkpoint = {
            "epoch": epoch, "model_state_dict": self.state_dict(),
            "num_properties": self.num_properties, "num_classes": self.num_classes,
            "input_channels": self.input_channels, "training_stage": getattr(self, "training_stage", 2),
        }
        if optimizer: checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if metrics: checkpoint["metrics"] = metrics
        import torch
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "PropertyYOLO":
        import torch
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_properties=checkpoint.get("num_properties", 11),
            num_classes=checkpoint.get("num_classes", 6),
            input_channels=checkpoint.get("input_channels", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.training_stage = checkpoint.get("training_stage", 1)
        return model