import cv2
import numpy as np
import base64
import os
import random
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tip_projector import BeerLambertTIPProjector
from routers.inference import predict_image, mat_to_base64_data_uri

router = APIRouter(prefix="/api")
projector = BeerLambertTIPProjector()

# Define cross-platform dynamic paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # e.g., backend/routers
BACKEND_DIR = os.path.dirname(CURRENT_DIR) # e.g., backend/
PROJECT_ROOT = os.path.dirname(BACKEND_DIR) # e.g., Root project folder

SAFE_DIR = os.path.join(BACKEND_DIR, "assets", "dataset", "safe")
THREATS_DIR = os.path.join(PROJECT_ROOT, "dataset", "splits", "test", "images")

class TIPRequest(BaseModel):
    bg_type: str
    threat_type: str
    scale: float = 1.0
    angle: float = 0.0
    pos_x: float = 50.0
    pos_y: float = 50.0
    material_thickness: float = 1.0

def get_blank_image(h=400, w=600, color=(255, 255, 255)):
    return np.full((h, w, 3), color, dtype=np.uint8)

def generate_mock_bg(mock_type: str) -> np.ndarray:
    if os.path.exists(SAFE_DIR):
        valid_files = [f for f in os.listdir(SAFE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if valid_files:
            filename = random.choice(valid_files)
            filepath = os.path.join(SAFE_DIR, filename)
            img = cv2.imread(filepath)
            if img is not None:
                return img
    print(f"⚠ Warning: No valid background images found in {SAFE_DIR}. Using fallback.")
    return get_blank_image(400, 600, (240, 240, 240))

def generate_threat_preset(threat_type: str) -> np.ndarray:
    if os.path.exists(THREATS_DIR):
        valid_files = [f for f in os.listdir(THREATS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if valid_files:
            filename = random.choice(valid_files)
            filepath = os.path.join(THREATS_DIR, filename)
            img = cv2.imread(filepath)
            if img is not None:
                return img
    print(f"⚠ Warning: No valid threat images found in {THREATS_DIR}. Using fallback.")
    return get_blank_image(150, 150, (255, 255, 255))

@router.post("/tip/project")
def run_tip_project(req: TIPRequest):
    try:
        bg_img = generate_mock_bg(req.bg_type)
        fg_img = generate_threat_preset(req.threat_type)

        projected_img, projection_bbox = projector.project_threat(
            bg_image=bg_img,
            fg_image=fg_img,
            scale=req.scale,
            angle_deg=req.angle,
            pos_x_pct=req.pos_x,
            pos_y_pct=req.pos_y,
            thickness=req.material_thickness
        )

        results = predict_image(projected_img, db_session=None)
        
        annotated_img = projected_img.copy()
        for bbox in results["bboxes"]:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            threat_level = bbox["threat_level"]
            color = (0, 0, 255) if threat_level == "CRITICAL" else ((0, 165, 255) if threat_level == "WARNING" else (0, 200, 0))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            lbl = f"#{bbox['id']+1} {bbox['material'].upper()}"
            cv2.putText(annotated_img, lbl, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return {
            "projection_bbox": projection_bbox,
            "scan_results": results,
            "composite_image": mat_to_base64_data_uri(annotated_img)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feed")
def get_luggage_feed():
    feed = [
        {
            "id": 1,
            "name": "Standard Carry-on (Safe)",
            "description": "Suitcase with organic clothes, toiletries, and some charger cables.",
            "threat_probability": 0.04,
            "expected_level": "SAFE",
            "mock_type": "safe_luggage"
        },
        {
            "id": 2,
            "name": "Heavy Metallic Toolbox (Warning)",
            "description": "Heavy metal toolbox containing wrenches, nails, and batteries.",
            "threat_probability": 0.42,
            "expected_level": "WARNING",
            "mock_type": "toolbox"
        },
        {
            "id": 3,
            "name": "Backpack with concealed Knife (Critical)",
            "description": "Fabric backpack containing books, clothing, and an elongated metal object (hunting knife).",
            "threat_probability": 0.88,
            "expected_level": "CRITICAL",
            "mock_type": "knife_bag"
        },
        {
            "id": 4,
            "name": "Suitcase with aerosol containers (Warning)",
            "description": "Suitcase with aerosol deodorant canisters and toiletries.",
            "threat_probability": 0.38,
            "expected_level": "WARNING",
            "mock_type": "aerosol_bag"
        },
        {
            "id": 5,
            "name": "Shielded Core Luggage (Critical)",
            "description": "Suspicious luggage containing a dark, rectangular lead-shielded box that blocks X-rays.",
            "threat_probability": 0.94,
            "expected_level": "CRITICAL",
            "mock_type": "shielded_bag"
        }
    ]
    return feed

@router.get("/feed/image/{mock_type}")
def get_mock_image(mock_type: str):
    img = generate_mock_bg(mock_type)
    _, buffer = cv2.imencode(".jpg", img)
    return JSONResponse(content={"image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"})
