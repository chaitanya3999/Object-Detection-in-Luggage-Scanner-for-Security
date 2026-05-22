"""
Model 2 Trainer (Random Forest / XGBoost)

Trains an explainable threat classifier using the 11-dimensional property vectors extracted from Model 1.
Handles data loading, train/test splitting, class imbalance (via class weights), and model evaluation.

Saves the trained model to checkpoints/model2.joblib and generates feature importance plots.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config

def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Model 2 Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(importances, features, output_path):
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Model 2 Property Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center", color='skyblue')
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Model 2 Threat Classifier")
    parser.add_argument("--csv", default="data/annotations/model2_dataset.csv", help="Path to generated property CSV")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--model", default="rf", choices=["rf", "xgb"], help="Algorithm to use")
    parser.add_argument("--output", default="checkpoints/model2.joblib", help="Output model path")
    args = parser.parse_args()

    config = load_config(args.config)
    
    print(f"\n{'='*65}")
    print(f"  Model 2 Classifier Training")
    print(f"{'='*65}")
    
    if not os.path.exists(args.csv):
        print(f"Error: Dataset not found at {args.csv}")
        print("Please run `python src/model2/prepare_dataset.py` first to generate the dataset.")
        sys.exit(1)

    print(f"Loading dataset from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Feature columns (the 11 properties)
    feature_cols = [
        "edge_sharpness", "length_to_width_ratio", "symmetry_score", 
        "curvature_index", "approximate_volume", "material_category", 
        "avg_absorption_intensity", "material_homogeneity", "density_level", 
        "sharp_edge_count", "occlusion_score"
    ]
    
    X = df[feature_cols]
    # We will predict the exact class label names to be robust
    y = df["label"]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class Distribution:\n{y.value_counts()}")
    
    # Stratified split to ensure rare threats are represented in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_test)} samples.")
    
    # Initialize classifier
    if args.model == "xgb" and HAS_XGB:
        print("Using XGBoost Classifier...")
        # XGBoost requires label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            # handle class imbalance
            scale_pos_weight=1.0 
        )
        clf.fit(X_train, y_train_enc)
        
        y_pred_enc = clf.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        importances = clf.feature_importances_
        classes = le.classes_
        
        # Save the label encoder with the model for inference
        model_payload = {
            "model": clf,
            "encoder": le,
            "features": feature_cols
        }
        
    else:
        if args.model == "xgb" and not HAS_XGB:
            print("⚠ XGBoost not installed. Falling back to Random Forest.")
            
        print("Using Random Forest Classifier...")
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',  # Handles class imbalance beautifully
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        importances = clf.feature_importances_
        classes = clf.classes_
        
        model_payload = {
            "model": clf,
            "encoder": None,
            "features": feature_cols
        }

    # Evaluate
    print("\n--- Evaluation Results ---")
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model_payload, args.output)
    print(f"\n✓ Saved trained Model 2 pipeline to {args.output}")
    
    # Visualizations
    results_dir = os.path.join(os.path.dirname(os.path.dirname(args.output)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plot_confusion_matrix(cm, classes, os.path.join(results_dir, "model2_confusion_matrix.png"))
    plot_feature_importance(importances, feature_cols, os.path.join(results_dir, "model2_feature_importance.png"))
    
    print(f"✓ Saved visual evaluation metrics to {results_dir}/")
    print(f"\nTop 3 Discriminative Features:")
    indices = np.argsort(importances)[::-1]
    for i in range(3):
        print(f"  {i+1}. {feature_cols[indices[i]]} ({importances[indices[i]]:.3f})")

if __name__ == "__main__":
    main()
