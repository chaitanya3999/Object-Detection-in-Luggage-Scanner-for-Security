import json
from backend.api import XRayAPI

api = XRayAPI(stage1_weights="checkpoints/best.pt", stage2_weights="checkpoints/stage2_ultimate.pth", device="cpu")
img_path = "runs/detect/train/train_batch0.jpg"
res = api.scan_luggage(img_path)
print(res)
