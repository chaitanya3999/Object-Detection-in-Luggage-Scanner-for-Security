"""
Visualization utilities for X-Ray Object Detection project.
Provides plotting functions for detections, training curves, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple


def visualize_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[str],
    property_vectors: Optional[List[Dict[str, float]]] = None,
    scores: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Visualize detected objects on an X-ray image with bounding boxes and property vectors.

    Args:
        image: Input image (H, W) or (H, W, C).
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        labels: List of class label strings.
        property_vectors: Optional list of property dicts for each detection.
        scores: Optional confidence scores for each detection.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display image
    if len(image.shape) == 2:
        ax.imshow(image, cmap="gray")
    elif image.shape[2] == 4:
        # 4-channel: show first 3 as RGB
        ax.imshow(image[:, :, :3])
    else:
        ax.imshow(image)

    # Color map for classes
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(set(labels)), 1)))
    unique_labels = list(set(labels))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = color_map[label]

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Build label text
        text = label
        if scores is not None and i < len(scores):
            text += f" ({scores[i]:.2f})"

        ax.text(
            x1, y1 - 5, text,
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
        )

        # Show property vector if available
        if property_vectors is not None and i < len(property_vectors):
            props = property_vectors[i]
            prop_text = "\n".join([f"{k}: {v:.3f}" for k, v in props.items()])
            ax.text(
                x2 + 5, y1, prop_text,
                color="white", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                verticalalignment="top",
            )

    ax.set_title("X-Ray Detection Results", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss/metric curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_metrics: Dict of metric_name -> values per epoch.
        val_metrics: Dict of metric_name -> values per epoch.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure.
    """
    num_plots = 1
    if train_metrics:
        num_plots += len(train_metrics)

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    # Loss curve
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metric curves
    if train_metrics:
        for i, (name, values) in enumerate(train_metrics.items()):
            ax = axes[i + 1]
            ax.plot(range(1, len(values) + 1), values, "b-", label=f"Train {name}", linewidth=2)
            if val_metrics and name in val_metrics:
                val_vals = val_metrics[name]
                ax.plot(range(1, len(val_vals) + 1), val_vals, "r-", label=f"Val {name}", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a confusion matrix with labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_property_distributions(
    property_vectors: np.ndarray,
    property_names: List[str],
    class_labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution histograms for each property dimension.

    Args:
        property_vectors: Array of shape (N, num_properties).
        property_names: Names for each property dimension.
        class_labels: Optional integer class labels for coloring.
        class_names: Optional class name strings.
        save_path: Optional save path.
    """
    num_props = len(property_names)
    cols = 4
    rows = (num_props + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(property_names, axes)):
        if class_labels is not None and class_names is not None:
            for cls_idx, cls_name in enumerate(class_names):
                mask = class_labels == cls_idx
                if mask.any():
                    ax.hist(property_vectors[mask, i], bins=30, alpha=0.6, label=cls_name)
            ax.legend(fontsize=6)
        else:
            ax.hist(property_vectors[:, i], bins=30, alpha=0.7, color="steelblue")

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for j in range(num_props, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Property Vector Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
