import cv2
import json
import base64
import numpy as np
import os
import sys
import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any

# Add root directory to sys.path to import api.py and backend.database
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

try:
    from backend.api import XRayAPI
    HAS_API = True
except ImportError as e:
    print(f"⚠ Failed to import XRayAPI (missing dependencies like torch?): {e}")
    HAS_API = False

from backend.database import get_db, ScanLog

router = APIRouter(prefix="/api")

# Initialize XRayAPI (Model 1) exactly ONCE globally on startup
print("Initializing Model 1 (XRayAPI) globally on startup...")
stage1_path = os.path.join(root_dir, "checkpoints", "best.pt")
stage2_path = os.path.join(root_dir, "checkpoints", "stage2_ultimate.pth")

try:
    if HAS_API:
        api_engine = XRayAPI(stage1_weights=stage1_path, stage2_weights=stage2_path, device="cpu")
        print("✓ Model 1 (XRayAPI) instantiated successfully.")
    else:
        api_engine = None
except Exception as e:
    print(f"⚠ Failed to instantiate XRayAPI: {e}")
    api_engine = None

# Initialize Model 2 (Random Forest) exactly ONCE globally on startup
print("Initializing Model 2 (Random Forest Classifier) globally...")
model2_path = os.path.join(root_dir, "checkpoints", "model2.joblib")
model2_payload = None
try:
    if os.path.exists(model2_path):
        model2_payload = joblib.load(model2_path)
        print("✓ Model 2 (Random Forest) loaded successfully.")
    else:
        print("⚠ Model 2 not found at backend/model2.joblib.")
except Exception as e:
    print(f"⚠ Failed to load Model 2: {e}")


def mat_to_base64_data_uri(image: np.ndarray, ext: str = ".jpg") -> str:
    _, buffer = cv2.imencode(ext, image)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def evaluate_threat_with_model2(props: dict) -> dict:
    """Uses Model 2 to classify threat based on Model 1's physical properties."""
    
    # MANUAL OVERRIDE: User requested ALL metallic objects be flagged as unsafe threats
    is_metallic = (props.get("material_category", 0) == 1) or (props.get("density_level", 0.0) > 0.45)
    
    if is_metallic:
        return {
            "level": "CRITICAL",
            "score": 0.99,
            "explanation": "Security Override: Sharp metallic object flagged as CRITICAL threat."
        }

    if model2_payload is not None:
        model = model2_payload["model"]
        encoder = model2_payload.get("encoder")
        features = model2_payload["features"]
        
        # Build feature vector
        vec = [props.get(f, 0.0) for f in features]
        X = pd.DataFrame([vec], columns=features)
        
        # Predict
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        if encoder is not None:
            class_idx = pred
            pred_label = encoder.inverse_transform([class_idx])[0]
            prob = probs[class_idx]
        else:
            pred_label = pred
            class_idx = list(model.classes_).index(pred_label)
            prob = probs[class_idx]
            
        if pred_label == "safe":
            return {
                "level": "SAFE",
                "score": float(prob),
                "explanation": f"Model 2: Classified as SAFE with {prob*100:.1f}% confidence."
            }
        elif pred_label in ["gun", "knife"]:
            return {
                "level": "CRITICAL",
                "score": float(prob),
                "explanation": f"Model 2: CRITICAL THREAT - {pred_label.upper()} with {prob*100:.1f}% confidence."
            }
        else:
            return {
                "level": "WARNING",
                "score": float(prob),
                "explanation": f"Model 2: WARNING - {pred_label.upper()} with {prob*100:.1f}% confidence."
            }
            
    # Fallback to simple heuristics if Model 2 is missing
    density = props.get("density_level", 0.0)
    if density > 0.75:
        return {"level": "CRITICAL", "score": 0.9, "explanation": "High density metallic mass detected."}
    elif density > 0.5:
        return {"level": "WARNING", "score": 0.6, "explanation": "Suspicious density detected."}
    return {"level": "SAFE", "score": 0.1, "explanation": "Properties within safe limits."}


@router.get("/model-status")
def get_model_status():
    return {
        "is_dl_mode": api_engine is not None,
        "device": getattr(api_engine, "device", "cpu") if api_engine else "cpu",
        "checkpoint_found": os.path.exists(stage1_path),
        "checkpoint_path": stage1_path,
        "model2_found": model2_payload is not None,
        "properties_schema": [
            {"index": 0, "name": "edge_sharpness", "type": "float [0,1]"},
            {"index": 1, "name": "length_to_width_ratio", "type": "float [0,10]"},
            {"index": 2, "name": "symmetry_score", "type": "float [0,1]"},
            {"index": 3, "name": "curvature_index", "type": "float [0,1]"},
            {"index": 4, "name": "approximate_volume", "type": "float [0,1]"},
            {"index": 5, "name": "material_category", "type": "int {0: Organic, 1: Metallic, 2: Mixed, 3: Opaque}"},
            {"index": 6, "name": "avg_absorption_intensity", "type": "float [0,1]"},
            {"index": 7, "name": "material_homogeneity", "type": "float [0,1]"},
            {"index": 8, "name": "density_level", "type": "float [0,1]"},
            {"index": 9, "name": "sharp_edge_count", "type": "int [0,20]"},
            {"index": 10, "name": "occlusion_score", "type": "float [0,1] (Computed Geometrically)"}
        ]
    }

try:
    from backend.inference import LuggageInferenceEngine
    fallback_engine = LuggageInferenceEngine()
except ImportError as e:
    print(f"⚠ Failed to import LuggageInferenceEngine: {e}")
    fallback_engine = None

def predict_image(img: np.ndarray, db_session: Session = None) -> dict:
    if not api_engine:
        if fallback_engine:
            print("⚠ Using CV Fallback Engine because XRayAPI is unavailable.")
            res = fallback_engine.predict(img)
            
            if db_session:
                threat_objects = [b for b in res["bboxes"] if b["threat_level"] in ["CRITICAL", "WARNING"]]
                log_entry = ScanLog(
                    overall_threat=res["overall"]["level"],
                    num_objects=len(res["bboxes"]),
                    threat_details=json.dumps(threat_objects),
                    inference_mode=res["mode"]
                )
                db_session.add(log_entry)
                db_session.commit()
                db_session.refresh(log_entry)
                res["scan_id"] = log_entry.id
            else:
                res["scan_id"] = None
                
            return res
        else:
            raise RuntimeError("XRayAPI Engine is not initialized and CV Fallback failed to load.")
        
    temp_dir = os.path.join(root_dir, "backend", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "temp_inference.jpg")
    cv2.imwrite(temp_path, img)
    
    raw_result_json = api_engine.scan_luggage(temp_path)
    api_result = json.loads(raw_result_json)
    
    if "error" in api_result:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise RuntimeError(api_result["error"])
        
    bboxes = []
    properties = {}
    max_threat_score = 0.0
    overall_level = "SAFE"
    
    detections = api_result.get("detections", [])
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bounding_box"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        props = det["physical_properties"]
        mapped_props = {
            "density_level": props.get("density_level", 0.0),
            "edge_sharpness": props.get("edge_sharpness", 0.0),
            "symmetry_score": props.get("symmetry_score", 0.0),
            "length_to_width_ratio": props.get("length_width_ratio", 0.0),
            "curvature_index": props.get("curvature_index", 0.0),
            "occlusion_score": props.get("occlusion_score", 0.0),
            "material_category": 1 if "metal" in det["material_signature"].lower() else 0,
            "approximate_volume": 0.0,
            "avg_absorption_intensity": 0.0,
            "material_homogeneity": 0.0,
            "sharp_edge_count": 0
        }
        properties[str(i)] = mapped_props
        
        threat_data = evaluate_threat_with_model2(mapped_props)
        threat_level = threat_data["level"]
        
        if threat_level == "CRITICAL":
            overall_level = "CRITICAL"
        elif threat_level == "WARNING" and overall_level != "CRITICAL":
            overall_level = "WARNING"
            
        material = det["material_signature"].lower()
        if "metal" in material: material = "metallic"
        elif "organic" in material: material = "organic"
        elif "plastic" in material: material = "mixed"
        else: material = "opaque"
        
        bboxes.append({
            "id": i,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "threat_level": threat_level,
            "material": material,
            "confidence": det["confidence_score"],
            "explanation": threat_data["explanation"]
        })
        
    if db_session:
        threat_objects = [b for b in bboxes if b["threat_level"] in ["CRITICAL", "WARNING"]]
        log_entry = ScanLog(
            overall_threat=overall_level,
            num_objects=len(bboxes),
            threat_details=json.dumps(threat_objects),
            inference_mode="Model 1 (DL) + Model 2 (RF)"
        )
        db_session.add(log_entry)
        db_session.commit()
        db_session.refresh(log_entry)
        scan_id = log_entry.id
    else:
        scan_id = None
        
    if os.path.exists(temp_path): os.remove(temp_path)
    
    return {
        "scan_id": scan_id,
        "mode": "Dual-Model Pipeline (YOLOv8 + RF)",
        "bboxes": bboxes,
        "properties": properties,
        "overall": {
            "level": overall_level,
            "classification": "Threat Detected" if overall_level != "SAFE" else "Clear",
            "explanation": f"Integrated Model 1 & 2 Diagnostics: {json.dumps(api_result.get('diagnostics', {}))}"
        }
    }

@router.post("/scan")
async def scan_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Receives an image, calls Model 1 (XRayAPI), evaluates properties with Model 2, and maps to UI format."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = predict_image(img, db)
        
        annotated_img = img.copy()
        for bbox in results["bboxes"]:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = (0, 0, 255) if bbox["threat_level"] == "CRITICAL" else ((0, 165, 255) if bbox["threat_level"] == "WARNING" else (0, 200, 0))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            lbl = f"#{bbox['id']+1} {bbox['material'].upper()}"
            cv2.putText(annotated_img, lbl, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        results["original_image"] = mat_to_base64_data_uri(img)
        results["annotated_image"] = mat_to_base64_data_uri(annotated_img)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failure: {str(e)}")
