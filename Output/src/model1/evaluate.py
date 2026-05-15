"""
Evaluation metrics for Model 1.

Detection:
  - mAP@0.5 and mAP@0.5:0.95

Property Regression:
  - Per-property MAE and RMSE
  - Property vector consistency (cosine similarity)

Material Classification:
  - Accuracy, confusion matrix
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class Model1Evaluator:
    """
    Comprehensive evaluator for Model 1 (Property Extraction).

    Usage:
        evaluator = Model1Evaluator(property_names, material_names)
        evaluator.update(predictions, targets)
        results = evaluator.compute()
    """

    def __init__(
        self,
        property_names: List[str] = None,
        material_names: List[str] = None,
        num_properties: int = 10,
        num_materials: int = 4,
    ):
        self.property_names = property_names or [
            "edge_sharpness", "length_width_ratio", "symmetry",
            "curvature", "volume", "material_cat", "absorption",
            "homogeneity", "density", "sharp_edges",
        ]
        self.material_names = material_names or [
            "organic", "metallic", "mixed", "opaque",
        ]
        self.num_properties = num_properties
        self.num_materials = num_materials

        self.reset()

    def reset(self):
        """Reset all accumulated predictions."""
        self.all_pred_properties = []
        self.all_gt_properties = []
        self.all_pred_materials = []
        self.all_gt_materials = []
        self.all_class_labels = []

    def update(
        self,
        pred_properties: np.ndarray,
        gt_properties: np.ndarray,
        pred_materials: Optional[np.ndarray] = None,
        gt_materials: Optional[np.ndarray] = None,
        class_labels: Optional[np.ndarray] = None,
    ):
        """
        Accumulate predictions and targets.

        Args:
            pred_properties: (N, num_properties) predicted property vectors.
            gt_properties: (N, num_properties) ground truth property vectors.
            pred_materials: (N,) predicted material class indices.
            gt_materials: (N,) ground truth material class indices.
            class_labels: (N,) object class labels for consistency analysis.
        """
        self.all_pred_properties.append(pred_properties)
        self.all_gt_properties.append(gt_properties)

        if pred_materials is not None:
            self.all_pred_materials.append(pred_materials)
        if gt_materials is not None:
            self.all_gt_materials.append(gt_materials)
        if class_labels is not None:
            self.all_class_labels.append(class_labels)

    def compute(self) -> Dict:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary with all metrics.
        """
        results = {}

        # Stack all predictions
        pred_props = np.concatenate(self.all_pred_properties, axis=0)
        gt_props = np.concatenate(self.all_gt_properties, axis=0)

        # Property regression metrics
        results["property"] = self._compute_property_metrics(pred_props, gt_props)

        # Material classification metrics
        if self.all_pred_materials and self.all_gt_materials:
            pred_mats = np.concatenate(self.all_pred_materials)
            gt_mats = np.concatenate(self.all_gt_materials)
            results["material"] = self._compute_material_metrics(pred_mats, gt_mats)

        # Property vector consistency
        if self.all_class_labels:
            class_labels = np.concatenate(self.all_class_labels)
            results["consistency"] = self._compute_consistency(pred_props, class_labels)

        return results

    def _compute_property_metrics(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> Dict:
        """Compute per-property MAE and RMSE."""
        metrics = {}

        # Skip categorical material property (index 5) from continuous metrics
        continuous_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9]

        # Overall MAE and RMSE
        errors = pred[:, continuous_idx] - gt[:, continuous_idx]
        metrics["overall_mae"] = float(np.abs(errors).mean())
        metrics["overall_rmse"] = float(np.sqrt((errors ** 2).mean()))

        # Per-property metrics
        metrics["per_property"] = {}
        for i, idx in enumerate(continuous_idx):
            name = self.property_names[idx] if idx < len(self.property_names) else f"prop_{idx}"
            prop_errors = pred[:, idx] - gt[:, idx]
            metrics["per_property"][name] = {
                "mae": float(np.abs(prop_errors).mean()),
                "rmse": float(np.sqrt((prop_errors ** 2).mean())),
                "std": float(prop_errors.std()),
                "max_error": float(np.abs(prop_errors).max()),
            }

        return metrics

    def _compute_material_metrics(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> Dict:
        """Compute material classification metrics."""
        metrics = {}
        metrics["accuracy"] = float((pred == gt).mean())

        # Confusion matrix
        cm = np.zeros((self.num_materials, self.num_materials), dtype=int)
        for p, g in zip(pred, gt):
            if 0 <= int(p) < self.num_materials and 0 <= int(g) < self.num_materials:
                cm[int(g), int(p)] += 1
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class precision, recall, F1
        metrics["per_class"] = {}
        for i, name in enumerate(self.material_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            metrics["per_class"][name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

        return metrics

    def _compute_consistency(
        self, pred_props: np.ndarray, class_labels: np.ndarray
    ) -> Dict:
        """
        Compute property vector consistency within each class.
        Same-class objects should have similar property vectors.
        """
        metrics = {}
        unique_classes = np.unique(class_labels)

        intra_class_similarities = []
        for cls in unique_classes:
            mask = class_labels == cls
            cls_vectors = pred_props[mask]

            if len(cls_vectors) < 2:
                continue

            # Compute pairwise cosine similarity
            norms = np.linalg.norm(cls_vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normalized = cls_vectors / norms

            # Mean pairwise similarity
            sim_matrix = normalized @ normalized.T
            n = len(sim_matrix)
            # Exclude diagonal
            mask_triu = np.triu_indices(n, k=1)
            similarities = sim_matrix[mask_triu]

            if len(similarities) > 0:
                intra_class_similarities.append(similarities.mean())

        if intra_class_similarities:
            metrics["mean_intra_class_similarity"] = float(np.mean(intra_class_similarities))
            metrics["std_intra_class_similarity"] = float(np.std(intra_class_similarities))
        else:
            metrics["mean_intra_class_similarity"] = 0.0
            metrics["std_intra_class_similarity"] = 0.0

        return metrics

    def print_results(self, results: Dict):
        """Pretty-print evaluation results."""
        print(f"\n{'='*60}")
        print("  Model 1 Evaluation Results")
        print(f"{'='*60}")

        # Property regression
        if "property" in results:
            prop = results["property"]
            print(f"\n📐 Property Regression:")
            print(f"   Overall MAE:  {prop['overall_mae']:.4f}")
            print(f"   Overall RMSE: {prop['overall_rmse']:.4f}")
            print(f"\n   Per-Property:")
            for name, m in prop["per_property"].items():
                status = "✓" if m["mae"] < 0.10 else "⚠"
                print(f"   {status} {name:25s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

        # Material classification
        if "material" in results:
            mat = results["material"]
            print(f"\n🧪 Material Classification:")
            status = "✓" if mat["accuracy"] > 0.92 else "⚠"
            print(f"   {status} Overall Accuracy: {mat['accuracy']:.4f}")
            print(f"\n   Per-Class:")
            for name, m in mat["per_class"].items():
                print(f"     {name:10s} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

        # Consistency
        if "consistency" in results:
            cons = results["consistency"]
            print(f"\n🔗 Property Vector Consistency:")
            print(f"   Mean intra-class cosine similarity: {cons['mean_intra_class_similarity']:.4f}")

        print(f"\n{'='*60}")


def evaluate_model1(
    model,
    test_loader,
    property_names: List[str] = None,
    device: str = "cpu",
) -> Dict:
    """
    Convenience function to evaluate Model 1.

    Args:
        model: Trained PropertyYOLO model.
        test_loader: Test DataLoader.
        property_names: Property names for reporting.
        device: Device string.

    Returns:
        Evaluation results dictionary.
    """
    evaluator = Model1Evaluator(property_names=property_names)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            gt_properties = batch["properties"]

            # Get predictions
            predictions = model.forward_properties(images)
            pred_properties = predictions["properties"].cpu().numpy()

            # Concatenate ground truth
            gt_props = torch.cat([p for p in gt_properties if len(p) > 0], dim=0).numpy()

            if len(gt_props) == 0:
                continue

            # Material predictions
            pred_materials = None
            gt_materials = None
            if "material_logits" in predictions:
                pred_materials = predictions["material_logits"].argmax(dim=1).cpu().numpy()
                gt_materials = gt_props[:, 5].astype(int)

            evaluator.update(
                pred_properties[:len(gt_props)],
                gt_props,
                pred_materials,
                gt_materials,
            )

    results = evaluator.compute()
    evaluator.print_results(results)
    return results
