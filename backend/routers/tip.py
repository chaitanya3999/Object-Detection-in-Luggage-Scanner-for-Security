import cv2
import numpy as np
import base64
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Adjust imports to be from the current dir since it's run from backend root
from tip_projector import BeerLambertTIPProjector
# We need the engine from inference router to keep singleton pattern, or initialize it.
# Let's import it from inference router to reuse it, or instantiate again (singleton is better).
from routers.inference import engine, mat_to_base64_data_uri

router = APIRouter(prefix="/api")

projector = BeerLambertTIPProjector()

@router.post("/tip")
async def threat_image_projection(
    bg_file: UploadFile = File(...),
    fg_file: UploadFile = File(...),
    scale: float = Form(1.0),
    angle: float = Form(0.0),
    pos_x_pct: float = Form(50.0),
    pos_y_pct: float = Form(50.0),
    thickness: float = Form(1.0)
):
    """
    Overlays a threat object on a suitcase using Beer-Lambert multiplying projection,
    then runs the threat detection engine on the synthetic image.
    """
    try:
        # Load background suitcase
        bg_contents = await bg_file.read()
        bg_arr = np.frombuffer(bg_contents, np.uint8)
        bg_img = cv2.imdecode(bg_arr, cv2.IMREAD_COLOR)
        
        # Load foreground threat
        fg_contents = await fg_file.read()
        fg_arr = np.frombuffer(fg_contents, np.uint8)
        fg_img = cv2.imdecode(fg_arr, cv2.IMREAD_COLOR)
        
        if bg_img is None or fg_img is None:
            raise HTTPException(status_code=400, detail="Invalid background or threat image.")
            
        # Perform Beer-Lambert projection overlay
        projected_img, projection_bbox = projector.project_threat(
            bg_image=bg_img,
            fg_image=fg_img,
            scale=scale,
            angle_deg=angle,
            pos_x_pct=pos_x_pct,
            pos_y_pct=pos_y_pct,
            thickness=thickness
        )
        
        # Run inference on the PROJECTED suitcase
        results = engine.predict(projected_img)
        
        # Annotate
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
            "bboxes": results["bboxes"],
            "properties": results["properties"],
            "overall": results["overall"],
            "projected_image": mat_to_base64_data_uri(projected_img),
            "annotated_image": mat_to_base64_data_uri(annotated_img)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TIP generation failed: {str(e)}")

@router.get("/feed")
def get_luggage_feed():
    """Returns a list of pre-configured simulation suitcases with varying threat degrees."""
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
    """Draws realistic simulated dual-energy X-ray luggage profiles programmatically and returns the JPEG."""
    h, w = 400, 600
    img = np.full((h, w, 3), (200, 235, 255), dtype=np.uint8) # Light soft beige-yellow
    
    cv2.rectangle(img, (40, 40), (w-40, h-40), (140, 140, 120), 4) # Suitcase frame
    cv2.circle(img, (45, 45), 10, (140, 140, 120), -1)
    cv2.circle(img, (w-45, 45), 10, (140, 140, 120), -1)
    cv2.circle(img, (45, h-45), 10, (140, 140, 120), -1)
    cv2.circle(img, (w-45, h-45), 10, (140, 140, 120), -1)
    
    cv2.rectangle(img, (w//2-60, 15), (w//2+60, 40), (100, 100, 100), 3)

    if mock_type == "safe_luggage":
        cv2.ellipse(img, (200, 180), (130, 80), 15, 0, 360, (40, 130, 220), -1) 
        cv2.ellipse(img, (380, 240), (140, 90), -20, 0, 360, (60, 150, 235), -1) 
        cv2.rectangle(img, (150, 240), (280, 320), (120, 190, 120), -1)
        cv2.ellipse(img, (180, 300), (60, 40), 45, 0, 180, (100, 160, 100), 2)
        cv2.ellipse(img, (240, 280), (40, 30), -30, 0, 270, (100, 160, 100), 2)

    elif mock_type == "toolbox":
        cv2.rectangle(img, (120, 150), (320, 175), (220, 120, 20), -1)
        cv2.circle(img, (120, 162), 22, (220, 120, 20), -1)
        cv2.circle(img, (120, 162), 10, (200, 235, 255), -1) 
        cv2.circle(img, (320, 162), 22, (220, 120, 20), -1)
        cv2.circle(img, (320, 162), 10, (200, 235, 255), -1)
        cv2.rectangle(img, (350, 120), (500, 250), (200, 90, 10), -1)
        cv2.line(img, (60, 270), (w-60, 270), (120, 180, 120), 4)
        cv2.rectangle(img, (100, 290), (260, 350), (50, 140, 210), -1)

    elif mock_type == "knife_bag":
        cv2.ellipse(img, (w//2, h//2+20), (180, 120), 0, 0, 360, (50, 130, 220), -1)
        blade_pts = np.array([[220, 200], [380, 185], [360, 220]], np.int32)
        cv2.fillPoly(img, [blade_pts], (230, 110, 15))
        cv2.rectangle(img, (160, 195), (220, 210), (100, 150, 100), -1)
        cv2.circle(img, (155, 202), 6, (100, 150, 100), -1)

    elif mock_type == "aerosol_bag":
        cv2.rectangle(img, (180, 130), (240, 260), (210, 110, 20), -1) 
        cv2.rectangle(img, (190, 150), (230, 250), (100, 180, 100), -1) 
        cv2.ellipse(img, (210, 130), (30, 15), 0, 0, 360, (210, 110, 20), -1) 
        cv2.rectangle(img, (320, 180), (380, 310), (210, 110, 20), -1)
        cv2.rectangle(img, (330, 200), (370, 300), (100, 180, 100), -1)
        cv2.ellipse(img, (350, 180), (30, 15), 0, 0, 360, (210, 110, 20), -1)
        cv2.ellipse(img, (130, 280), (80, 60), 10, 0, 360, (40, 120, 210), -1)

    elif mock_type == "shielded_bag":
        cv2.ellipse(img, (250, 200), (160, 110), 10, 0, 360, (50, 130, 220), -1)
        cv2.rectangle(img, (200, 130), (380, 250), (30, 30, 30), -1)
        cv2.circle(img, (200, 190), 40, (230, 100, 20), 2)
        cv2.circle(img, (380, 190), 40, (230, 100, 20), 2)
        
    _, buffer = cv2.imencode(".jpg", img)
    return JSONResponse(content={"image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"})
