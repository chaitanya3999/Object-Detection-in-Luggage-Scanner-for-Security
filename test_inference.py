import cv2
import numpy as np
from backend.inference import LuggageInferenceEngine

engine = LuggageInferenceEngine()
# Create a dummy image
img = np.ones((640, 640, 3), dtype=np.uint8) * 200
# Add a dark rectangle to simulate a dense object
cv2.rectangle(img, (100, 100), (200, 200), (50, 50, 50), -1)

try:
    res = engine.predict(img)
    print("SUCCESS")
    print(res)
except Exception as e:
    print("FAILED")
    import traceback
    traceback.print_exc()
