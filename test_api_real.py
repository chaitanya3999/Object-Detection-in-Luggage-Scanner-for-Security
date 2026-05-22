import os
import json
from backend.api import XRayAPI

api = XRayAPI(stage1_weights="checkpoints/best.pt", stage2_weights="checkpoints/stage2_ultimate.pth", device="cpu")
img_path = "dataset/splits/val/images/P07398_jpg.rf.c840a486735b75d8641c509f9c5b490b.jpg"

if not os.path.exists(img_path):
    print(f"Image not found at {img_path}")
else:
    res = api.scan_luggage(img_path)
    print(res)
