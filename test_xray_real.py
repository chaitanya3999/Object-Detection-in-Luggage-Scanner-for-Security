import cv2
import numpy as np

# Use a REAL X-ray image from the dataset
img = cv2.imread('dataset/processed/images/P01083_jpg.rf.2e88e0064ab094907d2747c67f9ac2e8.jpg')
if img is None:
    print("Not found")
    exit(1)

h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
b, g, r = cv2.split(img)

print(f"Blue  - min:{b.min()} max:{b.max()} mean:{b.mean():.1f}")
print(f"Green - min:{g.min()} max:{g.max()} mean:{g.mean():.1f}")
print(f"Red   - min:{r.min()} max:{r.max()} mean:{r.mean():.1f}")

# In X-ray scanners, metallic objects appear as blue/teal/green tinted regions
# Organic objects appear as orange/yellow

# Strategy: Find regions where Blue channel dominates over Red channel
# In orange areas: R >> B. In metallic blue areas: B >= R or B is high relative to R
metal_mask = np.zeros((h, w), dtype=np.uint8)

# Condition: blue is strong relative to red (blue/teal regions)
for thresh in [0, 10, 20, 30]:
    mask = ((b.astype(int) - r.astype(int)) > thresh).astype(np.uint8) * 255
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnt if 500 < cv2.contourArea(c) < w*h*0.8]
    print(f"\nB-R > {thresh}: {len(valid)} objects")

# Also try: green-dominant (green > red) for lighter metals
for thresh in [0, 10, 20]:
    mask = ((g.astype(int) - r.astype(int)) > thresh).astype(np.uint8) * 255
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnt if 500 < cv2.contourArea(c) < w*h*0.8]
    print(f"\nG-R > {thresh}: {len(valid)} objects")

# HSV approach - isolate non-orange regions
# Orange in HSV: H=5-25, high S
# Blue/teal: H=85-130
# Green: H=35-85
for lo_h, hi_h, name in [(80, 140, "blue"), (35, 85, "green"), (35, 140, "blue+green")]:
    mask = cv2.inRange(hsv, np.array([lo_h, 40, 40]), np.array([hi_h, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnt if 500 < cv2.contourArea(c) < w*h*0.8]
    print(f"\nHSV {name} (H={lo_h}-{hi_h}): {len(valid)} objects")
    for c in valid[:5]:
        x, y, bw, bh = cv2.boundingRect(c)
        print(f"  Box: ({x},{y},{bw},{bh}), Area: {cv2.contourArea(c):.0f}")

