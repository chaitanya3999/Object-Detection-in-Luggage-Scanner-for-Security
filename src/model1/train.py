"""
Training script for Model 1 (Property Extraction).

Two-stage training:
  Stage 1 (50-100 epochs): Detection only. Backbone from COCO pretrained weights.
  Stage 2 (30-50 epochs): Freeze backbone. Train property head + material branch.

Designed for Google Colab execution.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from .architecture import PropertyYOLO
from .loss import MultiTaskLoss


class Trainer:
    """
    Two-stage trainer for the PropertyYOLO model.

    Usage:
        trainer = Trainer(config)
        # Stage 1: Detection
        trainer.train_stage1(train_loader, val_loader)
        # Stage 2: Property regression
        trainer.train_stage2(train_loader, val_loader)
    """

    def __init__(
        self,
        config: dict,
        model: Optional[PropertyYOLO] = None,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.history = {"train_loss": [], "val_loss": [], "metrics": {}}

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"⚙ Using device: {self.device}")

        # Model
        if model is not None:
            self.model = model.to(self.device)
        else:
            model_cfg = config.get("model1", {})
            self.model = PropertyYOLO(
                model_size=model_cfg.get("backbone", "yolov8m"),
                num_classes=len(config.get("dataset", {}).get("threat_classes", ["threat"])) + 1,
                num_properties=model_cfg.get("property_head", {}).get("num_outputs", 11),
                input_channels=model_cfg.get("input_channels", 4),
                pretrained=model_cfg.get("pretrained", True),
                property_head_dims=model_cfg.get("property_head", {}).get("hidden_dims", [512, 256]),
                material_branch=model_cfg.get("material_branch", {}).get("enabled", True),
            ).to(self.device)

        # Loss
        loss_cfg = config.get("model1", {}).get("loss", {})
        self.criterion = MultiTaskLoss(
            use_uncertainty_weighting=loss_cfg.get("uncertainty_weighting", True),
            use_focal_loss=True,
            focal_gamma=loss_cfg.get("focal_loss", {}).get("gamma", 2.0),
            focal_alpha=loss_cfg.get("focal_loss", {}).get("alpha", 0.25),
            property_loss_type=loss_cfg.get("property_loss", "mse"),
        ).to(self.device)

        self.gradient_clip = loss_cfg.get("gradient_clip_max_norm", 1.0)

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize W&B
        if use_wandb:
            self._init_wandb(config)

    def _init_wandb(self, config: dict):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            wandb_cfg = config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "xray-detection"),
                entity=wandb_cfg.get("entity"),
                config=config,
            )
            self.wandb = wandb
        except ImportError:
            print("⚠ wandb not installed, disabling experiment tracking")
            self.use_wandb = False

    def train_stage1(
        self,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ):
        """
        Stage 1: Train detection only using YOLOv8's built-in training.

        This leverages Ultralytics' optimized training pipeline for
        detection, which handles the standard YOLO losses internally.
        """
        stage1_cfg = self.config.get("model1", {}).get("stage1", {})
        epochs = epochs or stage1_cfg.get("epochs", 100)
        data_yaml = self.config.get("data_yaml_path", "data/data.yaml")

        print(f"\n{'='*60}")
        print(f"  STAGE 1: Detection Training ({epochs} epochs)")
        print(f"{'='*60}")

        self.model.set_training_stage(1)

        # Use Ultralytics training for detection
        # This is the most efficient approach as it uses their optimized
        # training loop, data loading, and augmentation
        results = self.model.yolo.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.config.get("dataset", {}).get("image_size", 640),
            batch=stage1_cfg.get("batch_size", 16),
            lr0=stage1_cfg.get("learning_rate", 0.01),
            optimizer=stage1_cfg.get("optimizer", "SGD"),
            momentum=stage1_cfg.get("momentum", 0.937),
            weight_decay=stage1_cfg.get("weight_decay", 0.0005),
            warmup_epochs=stage1_cfg.get("warmup_epochs", 3),
            cos_lr=stage1_cfg.get("scheduler", "cosine") == "cosine",
            project=self.checkpoint_dir,
            name="stage1_detection",
            exist_ok=True,
            verbose=True,
        )

        # Save Stage 1 checkpoint
        self.model.save_checkpoint(
            os.path.join(self.checkpoint_dir, "stage1_best.pth"),
            epoch=epochs,
            metrics={"stage": 1},
        )

        print(f"✓ Stage 1 complete. Detection model saved.")
        return results

    def train_stage2(
        self,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ):
        """
        Stage 2: Train property regression head (backbone frozen).

        Custom training loop since Ultralytics doesn't support
        property regression natively.
        """
        stage2_cfg = self.config.get("model1", {}).get("stage2", {})
        epochs = epochs or stage2_cfg.get("epochs", 50)
        lr = lr or stage2_cfg.get("learning_rate", 0.001)

        print(f"\n{'='*60}")
        print(f"  STAGE 2: Property Regression Training ({epochs} epochs)")
        print(f"{'='*60}")

        self.model.set_training_stage(2)

        # Optimizer — only train property head parameters
        trainable_params = list(self.model.property_head.parameters())
        if self.model.material_branch is not None:
            trainable_params += list(self.model.material_branch.parameters())
        # Add uncertainty weight parameters
        trainable_params += list(self.criterion.parameters())

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=stage2_cfg.get("weight_decay", 0.0001),
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # Training
            train_loss = self._train_epoch_stage2(train_loader, optimizer, epoch)
            self.history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch_stage2(val_loader, epoch)
                self.history["val_loss"].append(val_loss)

            scheduler.step()

            # Logging
            lr_current = optimizer.param_groups[0]["lr"]
            log_msg = f"  Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            log_msg += f" | LR: {lr_current:.6f}"
            print(log_msg)

            if self.use_wandb:
                log_data = {"epoch": epoch, "train_loss": train_loss, "lr": lr_current}
                if val_loss is not None:
                    log_data["val_loss"] = val_loss
                self.wandb.log(log_data)

            # Save best model
            eval_loss = val_loss if val_loss is not None else train_loss
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                self.model.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "stage2_best.pth"),
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics={"val_loss": eval_loss},
                )

            # Save periodic checkpoints
            if epoch % 10 == 0:
                self.model.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"stage2_epoch{epoch}.pth"),
                    epoch=epoch,
                )

        print(f"✓ Stage 2 complete. Best val loss: {best_val_loss:.4f}")

    def _train_epoch_stage2(self, train_loader, optimizer, epoch: int) -> float:
        """Train one epoch for property regression."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            properties = batch["properties"]
            # Flatten properties for all objects in the batch
            gt_properties = torch.cat([p for p in properties if len(p) > 0], dim=0)

            if len(gt_properties) == 0:
                continue

            gt_properties = gt_properties.to(self.device)

            # Forward pass
            predictions = self.model.forward_properties(images)

            # Compute property regression loss
            targets = {"properties": gt_properties}

            # Material labels (from property index 5)
            if "material_logits" in predictions:
                material_labels = gt_properties[:, 5].long()
                targets["material_labels"] = material_labels

            loss, loss_dict = self.criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.get_trainable_params(),
                max_norm=self.gradient_clip,
            )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"    Batch {batch_idx} | Loss: {loss.item():.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate_epoch_stage2(self, val_loader, epoch: int) -> float:
        """Validate one epoch for property regression."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            images = batch["image"].to(self.device)
            properties = batch["properties"]
            gt_properties = torch.cat([p for p in properties if len(p) > 0], dim=0)

            if len(gt_properties) == 0:
                continue

            gt_properties = gt_properties.to(self.device)

            predictions = self.model.forward_properties(images)
            targets = {"properties": gt_properties}

            if "material_logits" in predictions:
                targets["material_labels"] = gt_properties[:, 5].long()

            loss, _ = self.criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def get_training_history(self) -> dict:
        """Return training history for plotting."""
        return self.history


def train_model1(
    config: dict,
    train_loader,
    val_loader=None,
    checkpoint_dir: str = "checkpoints",
):
    """
    Convenience function to train Model 1 end-to-end.

    Args:
        config: Configuration dictionary.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        checkpoint_dir: Directory to save checkpoints.
    """
    trainer = Trainer(
        config=config,
        checkpoint_dir=checkpoint_dir,
        use_wandb=config.get("wandb", {}).get("enabled", False),
    )

    # Stage 1: Detection
    trainer.train_stage1(train_loader, val_loader)

    # Stage 2: Property regression
    trainer.train_stage2(train_loader, val_loader)

    return trainer
