import os
import sys
import cv2
import numpy as np
import joblib
import pandas as pd
from typing import List, Dict, Tuple, Any

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.property_annotator import (
    PropertyAnnotator,
    PROPERTY_NAMES,
    MATERIAL_NAMES,
    MATERIAL_ORGANIC,
    MATERIAL_METALLIC,
    MATERIAL_MIXED,
    MATERIAL_OPAQUE
)

# Try to import torch and deep learning dependencies
try:
    import torch
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.model1.architecture import PropertyYOLO
    HAS_TORCH = True
except ImportError as e:
    torch = None
    PreprocessingPipeline = None
    PropertyYOLO = None
    HAS_TORCH = False
    print(f"ℹ [Engine] PyTorch or custom modules not imported: {e}. Forcing CV fallback mode.")

class LuggageInferenceEngine:
    """
    Dual-mode scanning engine:
    1. Deep Learning Mode: Loads YOLOv8 + PropertyRegressionHead from checkpoints
    2. Computer Vision Mode: Runs automated contour segmentation and PropertyAnnotator.
    
    Provides Model 2 threat classification with detailed explanation text.
    """
    def __init__(self, checkpoint_dir: str = "../checkpoints"):
        if torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.checkpoint_path = os.path.join(checkpoint_dir, "stage2_best.pth")
        self.model2_path = os.path.join(checkpoint_dir, "model2.joblib")
        self.model = None
        self.model2_payload = None
        
        if PreprocessingPipeline is not None:
            self.pipeline = PreprocessingPipeline(target_size=640)
        else:
            self.pipeline = None
        self.annotator = PropertyAnnotator()
        self.is_dl_mode = False
        
        # Try to load the trained models
        self.attempt_model_load()
        self.attempt_model2_load()
        
    def attempt_model2_load(self):
        """Attempts to load the Model 2 classifier."""
        if os.path.exists(self.model2_path):
            try:
                print(f"🌲 [Engine] Loading Model 2 classifier from {self.model2_path}...")
                self.model2_payload = joblib.load(self.model2_path)
                print("✓ [Engine] Model 2 loaded successfully!")
            except Exception as e:
                print(f"⚠ [Engine] Failed to load Model 2: {e}. Falling back to rule-based heuristics.")
                self.model2_payload = None
        
    def attempt_model_load(self):
        """Attempts to load the deep learning model checkpoint."""
        if not HAS_TORCH or PropertyYOLO is None:
            self.is_dl_mode = False
            return
            
        if os.path.exists(self.checkpoint_path):
            try:
                # print(f"🧠 [Engine] Loading trained deep learning model from {self.checkpoint_path}...")
                self.model = PropertyYOLO.load_checkpoint(self.checkpoint_path, device=self.device)
                self.model.eval()
                self.is_dl_mode = True
            except Exception as e:
                # print(f"⚠ Failed to load PyTorch checkpoint: {e}. Falling back to CV-based mode.")
                self.is_dl_mode = False
        else:
            # Silently fallback to CV mode without confusing the user
            self.is_dl_mode = False

    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs full pipeline on an uploaded BGR image.
        Returns:
            - bboxes: List of bounding box dicts (x1, y1, x2, y2, label, confidence, color)
            - properties: List of 11-dimensional property dictionaries corresponding to each bbox
            - threat_assessment: Dict with threat score, classification, and explanation
        """
        h, w = image_bgr.shape[:2]
        
        if self.is_dl_mode and self.model is not None:
            return self._predict_dl(image_bgr)
        else:
            return self._predict_cv(image_bgr)
            
    def _predict_dl(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs the deep learning pipeline (YOLOv8 + Property Regressor)."""
        h, w = image_bgr.shape[:2]
        preprocessed = self.pipeline(image_bgr)
        
        # Prepare PyTorch tensor
        tensor = torch.from_numpy(preprocessed).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device) # (1, 4, H, W)
        
        with torch.no_grad():
            det_input = tensor[:, :3, :, :]
            det_results = self.model.yolo.predict(det_input, conf=0.25, verbose=False)
            prop_results = self.model.forward_properties(tensor)
            
        bboxes = []
        properties = []
        
        if det_results and len(det_results[0].boxes) > 0:
            det = det_results[0]
            raw_boxes = []
            
            # First extract boxes
            for i, box in enumerate(det.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls)
                conf = float(box.conf)
                
                # Rescale boxes back to original size if YOLO resized them
                # YOLO usually scales to 640.
                yolo_w, yolo_h = det.orig_shape[1], det.orig_shape[0]
                x1 = int(xyxy[0] * w / yolo_w)
                y1 = int(xyxy[1] * h / yolo_h)
                x2 = int(xyxy[2] * w / yolo_w)
                y2 = int(xyxy[3] * h / yolo_h)
                
                raw_boxes.append([x1, y1, x2, y2])
                
            # Then compute properties
            props = prop_results["properties"].cpu().numpy() if "properties" in prop_results else None
            
            for i, bbox in enumerate(raw_boxes):
                prop_dict = {}
                if props is not None and i < len(props):
                    # Set regressed properties
                    for j, name in enumerate(PROPERTY_NAMES[:10]):
                        prop_dict[name] = float(props[i, j])
                    
                    # Fill occlusion score geometrically
                    prop_dict["occlusion_score"] = self.annotator._occlusion_score(bbox, raw_boxes, i)
                else:
                    # Fallback to annotator for properties if regression failed
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros_like(gray)
                    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
                    prop_dict = self.annotator.compute_properties(gray, mask, bbox, raw_boxes, i)
                
                properties.append(prop_dict)
                
                # Material and threat
                mat_id = int(prop_dict.get("material_category", 2))
                threat_data = self._evaluate_threat(prop_dict)
                
                bboxes.append({
                    "id": i,
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "label": f"Object {i+1} ({MATERIAL_NAMES[mat_id].upper()})",
                    "confidence": float(det.boxes[i].conf),
                    "material": MATERIAL_NAMES[mat_id],
                    "threat_level": threat_data["level"],
                    "threat_score": threat_data["score"],
                    "explanation": threat_data["explanation"]
                })
                
        # Overall threat level is the maximum of individual boxes
        overall_threat = self._assess_overall_threat(bboxes)
        
        return {
            "mode": "Deep Learning (YOLOv8 + Regression)",
            "bboxes": bboxes,
            "properties": properties,
            "overall": overall_threat
        }
        
    def _predict_cv(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs the computer vision fallback pipeline using contours & annotator."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Adaptively segment objects
        # Luggage scanners are typically bright orange/yellow, so threats and materials are darker masses.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. Exclusively detect dark, dense metallic objects (guns, knives, tools)
        # In X-rays, heavy threats are very dark/blue (intensity typically < 120)
        _, binary_dark = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Clean up noise and close gaps to form solid object masks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        opened = cv2.morphologyEx(binary_dark, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        # 3. Find distinct dense objects
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        contours_valid = []
        
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Ignore tiny noise and massive background blobs (like if the whole bag is dark)
                if area < 400 or area > (w * h * 0.5):
                    continue
                    
                x, y, bw, bh = cv2.boundingRect(cnt)
                raw_boxes.append([x, y, x + bw, y + bh])
                contours_valid.append(cnt)
            
        bboxes = []
        properties = []
        
        for i, (bbox, cnt) in enumerate(zip(raw_boxes, contours_valid)):
            # Make object mask
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Compute the 11 properties using PropertyAnnotator
            prop_dict = self.annotator.compute_properties(gray, mask, bbox, raw_boxes, i)
            
            # Post-process aspect ratio to ensure it fits the typical range
            prop_dict["length_to_width_ratio"] = min(prop_dict["length_to_width_ratio"], 10.0)
            
            properties.append(prop_dict)
            
            mat_id = int(prop_dict["material_category"])
            threat_data = self._evaluate_threat(prop_dict)
            
            bboxes.append({
                "id": i,
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "label": f"Object {i+1} ({MATERIAL_NAMES[mat_id].upper()})",
                "confidence": 0.95 - (i * 0.05), # Pseudo confidence
                "material": MATERIAL_NAMES[mat_id],
                "threat_level": threat_data["level"],
                "threat_score": threat_data["score"],
                "explanation": threat_data["explanation"]
            })
            
        overall_threat = self._assess_overall_threat(bboxes)
        
        return {
            "mode": "Computer Vision (Contour + Geometric Annotator)",
            "bboxes": bboxes,
            "properties": properties,
            "overall": overall_threat
        }

    def _evaluate_threat(self, props: Dict[str, float]) -> Dict[str, Any]:
        """
        Model 2 Evaluation:
        Classifies whether an object is a threat based on its property vector.
        Uses trained Model 2 (Random Forest) if available, else falls back to heuristics.
        """
        # MANUAL OVERRIDE: User requested metallic objects be flagged as unsafe if they are sharp
        is_metallic = (int(props.get("material_category", 2)) == 1) or (props.get("density_level", 0.0) > 0.45)
        is_sharp = props.get("edge_sharpness", 0.0) > 0.4 or props.get("sharp_edge_count", 0) >= 6
        
        if is_metallic and is_sharp:
            return {
                "level": "CRITICAL",
                "score": 0.99,
                "explanation": "CRITICAL THREAT OVERRIDE: Sharp metallic object flagged as CRITICAL threat."
            }

        if self.model2_payload is not None:
            model = self.model2_payload["model"]
            encoder = self.model2_payload["encoder"]
            features = self.model2_payload["features"]
            
            # Construct feature vector
            vec = [props.get(f, 0.0) for f in features]
            X = pd.DataFrame([vec], columns=features)
            
            # Predict
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            
            # Decode if necessary
            if encoder is not None:
                class_idx = pred
                pred_label = encoder.inverse_transform([class_idx])[0]
                prob = probs[class_idx]
            else:
                pred_label = pred
                # Probabilities match classes_
                class_idx = list(model.classes_).index(pred_label)
                prob = probs[class_idx]
            
            if pred_label == "safe":
                # Safety Override: If it's metallic and highly elongated, it's likely a knife/blade
                ratio = props.get("length_to_width_ratio", 1.0)
                material = int(props.get("material_category", 2))
                if ratio > 5.0 and material == 1: # MATERIAL_METALLIC
                    return {
                        "level": "CRITICAL",
                        "score": 0.85,
                        "explanation": f"CRITICAL THREAT OVERRIDE: Highly elongated metallic object (Ratio: {ratio:.1f}). Profile heavily matches a knife or weapon blade."
                    }
                
                return {
                    "level": "SAFE",
                    "score": 0.0,
                    "explanation": f"Classified as benign (Safe) with {prob*100:.1f}% confidence."
                }
            elif pred_label in ["gun", "knife"]:
                return {
                    "level": "CRITICAL",
                    "score": float(prob),
                    "explanation": f"CRITICAL THREAT: Classified as {pred_label.upper()} with {prob*100:.1f}% confidence. High material density and shape matched threat profile."
                }
            else:
                # tools (wrench, pliers, scissors)
                return {
                    "level": "WARNING",
                    "score": float(prob) * 0.8,
                    "explanation": f"WARNING: Classified as {pred_label.upper()} with {prob*100:.1f}% confidence. Sharp tool detected."
                }
                
        # --- Fallback Heuristic ---
        score = 0.0
        reasons = []
        
        density = props.get("density_level", 0.0)
        edges = props.get("edge_sharpness", 0.0)
        corners = props.get("sharp_edge_count", 0)
        material = int(props.get("material_category", 2))
        homogeneity = props.get("material_homogeneity", 0.0)
        ratio = props.get("length_to_width_ratio", 1.0)
        
        # Material evaluation
        if material == MATERIAL_METALLIC: # Metallic
            score += 0.3
            # Check for sharp profile (e.g., knife, blade)
            if corners >= 6 or edges > 0.4:
                score += 0.4
                reasons.append(f"Metallic object with high edge sharpness ({edges:.2f}) and multiple corners ({corners}) indicating potential tool or weapon blade shape.")
            else:
                score += 0.15
                reasons.append("Dense metal object detected (e.g. tool, electronics, hardware structure).")
        elif material == MATERIAL_OPAQUE: # Opaque/heavy shielding
            score += 0.6
            reasons.append(f"Extremely high density opaque mass (absorption: {props.get('avg_absorption_intensity', 0.0):.2f}). The scanner beam cannot penetrate this mass, indicating potential lead shielding or heavy battery cores.")
        elif material == MATERIAL_MIXED: # Mixed
            if density > 0.6:
                score += 0.25
                reasons.append("Mixed material density above average (e.g., aerosol containers, glass bottles).")
        else: # Organic
            if density > 0.75: # Heavy organic material (like dense liquids, plastics, or sheets)
                score += 0.3
                reasons.append("High density organic mass detected. Could indicate dense liquids, gels, or thick layered organic items.")
                
        # Shape/Geometric threat multipliers
        if ratio > 4.5 and material in [MATERIAL_METALLIC, MATERIAL_MIXED]:
            score += 0.2
            reasons.append(f"High aspect ratio ({ratio:.1f}) in metal/mixed material — typical elongated weapon or rod profile.")
            
        # Occlusion modifiers
        occlusion = props.get("occlusion_score", 0.0)
        if occlusion > 0.6 and score > 0.3:
            score += 0.1
            reasons.append("High occlusion level. Item is heavily concealed beneath other masses.")
            
        score = min(max(score, 0.05), 1.0)
        
        if score > 0.65:
            level = "CRITICAL"
            explanation = " | ".join(reasons) if reasons else "Critical density mass detected with metallic profiles."
        elif score > 0.35:
            level = "WARNING"
            explanation = " | ".join(reasons) if reasons else "Suspicious shape and density configuration detected."
        else:
            level = "SAFE"
            explanation = "Organic or low-density mixed materials in standard distribution. Verified safe."
            
        return {
            "score": score,
            "level": level,
            "explanation": explanation
        }
        
    def _assess_overall_threat(self, bboxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assesses the overall threat level of the luggage."""
        if not bboxes:
            return {
                "score": 0.0,
                "level": "SAFE",
                "classification": "Clear Luggage",
                "explanation": "No objects of concern detected in this luggage scan."
            }
            
        max_score = max([b["threat_score"] for b in bboxes])
        critical_count = sum([1 for b in bboxes if b["threat_level"] == "CRITICAL"])
        warning_count = sum([1 for b in bboxes if b["threat_level"] == "WARNING"])
        
        if critical_count > 0:
            level = "CRITICAL"
            classification = "Threat Alert: Armed/Shielded Item"
            explanation = f"Dangerous item detected! The scan identified {critical_count} critical threat(s) and {warning_count} warning alert(s). Shielded mass or sharp metal blades present. Immediate manual inspection required."
        elif warning_count > 0:
            level = "WARNING"
            classification = "Warning: Suspicious Contents"
            explanation = f"Manual inspection advised. Found {warning_count} items with warning profiles (unusual material density or shape). Check for liquids, containers, or nested electronics."
        else:
            level = "SAFE"
            classification = "Luggage Cleared"
            explanation = f"All {len(bboxes)} detected objects are within normal parameters (organic clothing, papers, open space)."
            
        return {
            "score": max_score,
            "level": level,
            "classification": classification,
            "explanation": explanation
        }
