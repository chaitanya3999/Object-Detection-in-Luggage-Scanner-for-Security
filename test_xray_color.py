import cv2
import numpy as np

# Test with the user's X-ray image
img_path = '/Users/chaitanya/.gemini/antigravity/brain/6de565d8-07a1-432e-8246-552c6ce6c2de/media__1773201601817.png'
img = cv2.imread(img_path)
if img is None:
    print("Image not found at that path, listing available media files...")
    import glob
    files = glob.glob('/Users/chaitanya/.gemini/antigravity/brain/6de565d8-07a1-432e-8246-552c6ce6c2de/media*')
    print(files)
    exit(1)

h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

# X-ray images use COLOR to distinguish materials:
# - Blue/dark blue/teal = METALLIC (guns, knives, tools)
# - Orange/yellow = ORGANIC (clothes, food, etc.)
# So we need to use HSV color space to isolate blue/teal regions

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
b, g, r = cv2.split(img)

print(f"\nChannel statistics:")
print(f"Blue  - min: {b.min()}, max: {b.max()}, mean: {b.mean():.1f}")
print(f"Green - min: {g.min()}, max: {g.max()}, mean: {g.mean():.1f}")
print(f"Red   - min: {r.min()}, max: {r.max()}, mean: {r.mean():.1f}")

# Method 1: Blue-dominant pixels (where blue > red and blue > green)
blue_dominant = ((b.astype(int) > r.astype(int) + 20) & (b > 50)).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
blue_clean = cv2.morphologyEx(blue_dominant, cv2.MORPH_CLOSE, kernel)
blue_clean = cv2.morphologyEx(blue_clean, cv2.MORPH_OPEN, kernel)
contours_blue, _ = cv2.findContours(blue_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_blue = [c for c in contours_blue if cv2.contourArea(c) > 500]
print(f"\nMethod 1 (Blue-dominant): {len(valid_blue)} objects")
for c in valid_blue:
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"  Box: ({x},{y},{bw},{bh}), Area: {cv2.contourArea(c):.0f}")

# Method 2: HSV blue/teal range (hue 85-130 for blue/teal in X-ray)
lower_blue = np.array([85, 30, 30])
upper_blue = np.array([135, 255, 255])
mask_blue_hsv = cv2.inRange(hsv, lower_blue, upper_blue)
mask_blue_hsv = cv2.morphologyEx(mask_blue_hsv, cv2.MORPH_CLOSE, kernel)
mask_blue_hsv = cv2.morphologyEx(mask_blue_hsv, cv2.MORPH_OPEN, kernel)
contours_hsv, _ = cv2.findContours(mask_blue_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_hsv = [c for c in contours_hsv if cv2.contourArea(c) > 500]
print(f"\nMethod 2 (HSV blue/teal): {len(valid_hsv)} objects")
for c in valid_hsv:
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"  Box: ({x},{y},{bw},{bh}), Area: {cv2.contourArea(c):.0f}")

# Method 3: Green/teal (for lighter metals)
lower_green = np.array([35, 30, 30])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_green = [c for c in contours_green if cv2.contourArea(c) > 500]
print(f"\nMethod 3 (HSV green/teal): {len(valid_green)} objects")
for c in valid_green:
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"  Box: ({x},{y},{bw},{bh}), Area: {cv2.contourArea(c):.0f}")

# Method 4: Combined blue+green (all metallic)
mask_metal = cv2.bitwise_or(mask_blue_hsv, mask_green)
mask_metal = cv2.morphologyEx(mask_metal, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
mask_metal = cv2.morphologyEx(mask_metal, cv2.MORPH_OPEN, kernel)
contours_metal, _ = cv2.findContours(mask_metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_metal = [c for c in contours_metal if cv2.contourArea(c) > 800]
print(f"\nMethod 4 (Combined metal mask): {len(valid_metal)} objects")
for c in valid_metal:
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"  Box: ({x},{y},{bw},{bh}), Area: {cv2.contourArea(c):.0f}")

