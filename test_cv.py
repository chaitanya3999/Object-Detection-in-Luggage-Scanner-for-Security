import cv2
import numpy as np
import os

img_path = 'backend/temp/temp_inference.jpg'
if not os.path.exists(img_path):
    print("Temp image not found!")
    exit(1)

img = cv2.imread(img_path)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 30, 150)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
valid = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 800 and area < (w * h * 0.5):
        valid.append(cnt)

print(f"Found {len(valid)} objects.")
for c in valid:
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"Box: {x, y, bw, bh}")

