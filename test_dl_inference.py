import cv2
import numpy as np
from backend.api import XRayAPI
from backend.routers.inference import evaluate_threat_with_model2

api = XRayAPI(stage1_weights="checkpoints/best.pt", stage2_weights="checkpoints/stage2_ultimate.pth")
img = np.ones((640, 640, 3), dtype=np.uint8) * 200
cv2.rectangle(img, (100, 100), (200, 200), (50, 50, 50), -1)

try:
    res = api.predict(img)
    print("SUCCESS")
    print(res)
except Exception as e:
    print("FAILED")
    import traceback
    traceback.print_exc()
