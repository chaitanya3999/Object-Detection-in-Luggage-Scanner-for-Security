import cv2
import json
import base64
import numpy as np
import os
import sys
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any

# Add root directory to sys.path to import api.py and backend.database
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from api import XRayAPI
from backend.database import get_db, ScanLog

router = APIRouter(prefix="/api")

# Initialize XRayAPI exactly ONCE globally on startup as requested
print("Initializing XRayAPI globally on startup...")
stage1_path = os.path.join(root_dir, "best.pt")
stage2_path = os.path.join(root_dir, "stage2_ultimate.pth")

try:
    api_engine = XRayAPI(stage1_weights=stage1_path, stage2_weights=stage2_path, device="cpu")
    print("✓ XRayAPI instantiated successfully.")
except Exception as e:
    print(f"⚠ Failed to instantiate XRayAPI: {e}")
    api_engine = None

def mat_to_base64_data_uri(image: np.ndarray, ext: str = ".jpg") -> str:
    _, buffer = cv2.imencode(ext, image)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

@router.get("/model-status")
def get_model_status():
    return {
        "is_dl_mode": api_engine is not None,
        "device": getattr(api_engine, "device", "cpu") if api_engine else "cpu",
        "checkpoint_found": os.path.exists(stage1_path),
        "checkpoint_path": stage1_path,
        "model2_found": os.path.exists(stage2_path),
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

@router.post("/scan")
async def scan_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Receives an uploaded luggage image, saves it to temp, calls XRayAPI, and maps to UI format."""
    if not api_engine:
        raise HTTPException(status_code=500, detail="XRayAPI Engine is not initialized.")
        
    try:
        contents = await file.read()
        
        # Save to temp folder as requested by the user
        temp_dir = os.path.join(root_dir, "backend", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename or "temp_upload.jpg")
        
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        # Call the requested XRayAPI method
        raw_result_json = api_engine.scan_luggage(temp_path)
        api_result = json.loads(raw_result_json)
        
        if "error" in api_result:
            raise HTTPException(status_code=500, detail=api_result["error"])
            
        # Read the image to draw bounding boxes for the UI
        img = cv2.imread(temp_path)
        annotated_img = img.copy()
        
        # Transform api_result into the format expected by the React frontend UI
        bboxes = []
        properties = {}
        
        detections = api_result.get("detections", [])
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bounding_box"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            threat_severity = det["threat_severity_index"]
            if threat_severity >= 85:
                threat_level = "CRITICAL"
                color = (0, 0, 255) # Red
            elif threat_severity >= 50:
                threat_level = "WARNING"
                color = (0, 165, 255) # Orange
            else:
                threat_level = "SAFE"
                mat = det["material_signature"]
                if "Organic" in mat: color = (0, 140, 255)
                elif "Metal" in mat: color = (255, 120, 0)
                elif "Plastic" in mat: color = (0, 200, 0)
                else: color = (60, 60, 60)
                
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
                "explanation": f"Classified as {det['classification']} with severity {threat_severity}/100."
            })
            
            # Map physical properties to the UI's radar chart
            props = det["physical_properties"]
            properties[str(i)] = {
                "density_level": props.get("density_level", 0),
                "edge_sharpness": props.get("edge_sharpness", 0),
                "symmetry_score": props.get("symmetry_score", 0),
                "length_to_width_ratio": props.get("length_width_ratio", 0),
                "curvature_index": props.get("curvature_index", 0),
                "occlusion_score": props.get("occlusion_score", 0),
                "material_category": 1 if "metal" in det["material_signature"].lower() else 0
            }
            
            # Draw bounding boxes and labels
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            lbl = f"#{i+1} {material.upper()} ({int(det['confidence_score']*100)}%)"
            cv2.putText(annotated_img, lbl, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Log to SQLite Database
        threat_objects = [b for b in bboxes if b["threat_level"] in ["CRITICAL", "WARNING"]]
        
        overall_level = "SAFE"
        protocol = api_result.get("security_protocol", "STANDARD_CLEARANCE")
        if "CRITICAL" in protocol: overall_level = "CRITICAL"
        elif "MANUAL" in protocol or "CAUTION" in protocol: overall_level = "WARNING"
            
        log_entry = ScanLog(
            overall_threat=overall_level,
            num_objects=len(bboxes),
            threat_details=json.dumps(threat_objects),
            inference_mode="Dual-Head XRayAPI"
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        original_base64 = mat_to_base64_data_uri(img)
        annotated_base64 = mat_to_base64_data_uri(annotated_img)
        
        # Cleanup temporary uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "scan_id": log_entry.id,
            "mode": "Dual-Head Deep Learning (XRayAPI)",
            "bboxes": bboxes,
            "properties": properties,
            "overall": {
                "level": overall_level,
                "classification": protocol.replace("_", " "),
                "explanation": f"API Composition: {api_result.get('diagnostics', {}).get('composition_breakdown', {})}"
            },
            "original_image": original_base64,
            "annotated_image": annotated_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failure: {str(e)}")
