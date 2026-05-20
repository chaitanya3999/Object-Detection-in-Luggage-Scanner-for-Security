import cv2
import json
import base64
import numpy as np
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any

# Adjust path imports if needed, but since it's run from backend root, imports from current dir work
from inference import LuggageInferenceEngine
from database import get_db, ScanLog

router = APIRouter(prefix="/api")

# We can initialize the engine here or inject it. Let's initialize a singleton for the router.
# Using the same path as main.py did
engine = LuggageInferenceEngine(checkpoint_dir="../checkpoints")

def mat_to_base64_data_uri(image: np.ndarray, ext: str = ".jpg") -> str:
    _, buffer = cv2.imencode(ext, image)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

@router.get("/model-status")
def get_model_status():
    return {
        "is_dl_mode": engine.is_dl_mode,
        "device": engine.device,
        "checkpoint_found": os.path.exists(engine.checkpoint_path),
        "checkpoint_path": os.path.abspath(engine.checkpoint_path),
        "model2_found": engine.model2_payload is not None,
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
    """Receives an uploaded luggage image, runs property extraction + threat evaluation, and returns data + annotated image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
            
        # Run prediction pipeline
        results = engine.predict(img)
        
        # Create an annotated version of the image to display bounding boxes on the frontend
        annotated_img = img.copy()
        for bbox in results["bboxes"]:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            threat_level = bbox["threat_level"]
            
            # Color coding based on threat level
            if threat_level == "CRITICAL":
                color = (0, 0, 255) # Red
            elif threat_level == "WARNING":
                color = (0, 165, 255) # Orange
            else:
                # Color coding based on material
                mat = bbox["material"]
                if mat == "organic":
                    color = (0, 140, 255) # Organic Orange in BGR
                elif mat == "metallic":
                    color = (255, 120, 0) # Metallic Blue in BGR
                elif mat == "mixed":
                    color = (0, 200, 0) # Mixed Green
                else:
                    color = (60, 60, 60) # Opaque Dark gray
                    
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            # Label
            lbl = f"#{bbox['id']+1} {bbox['material'].upper()} ({int(bbox.get('confidence', 0.95)*100)}%)"
            cv2.putText(annotated_img, lbl, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Log to Database!
        threat_objects = [b for b in results["bboxes"] if b["threat_level"] in ["CRITICAL", "WARNING"]]
        
        log_entry = ScanLog(
            overall_threat=results["overall"]["level"],
            num_objects=len(results["bboxes"]),
            threat_details=json.dumps(threat_objects),
            inference_mode=results["mode"]
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        # Generate base64 representations
        original_base64 = mat_to_base64_data_uri(img)
        annotated_base64 = mat_to_base64_data_uri(annotated_img)
        
        return {
            "scan_id": log_entry.id,
            "mode": results["mode"],
            "bboxes": results["bboxes"],
            "properties": results["properties"],
            "overall": results["overall"],
            "original_image": original_base64,
            "annotated_image": annotated_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failure: {str(e)}")
