import cv2
import numpy as np
import os

img_path = '/Users/chaitanya/.gemini/antigravity/brain/6de565d8-07a1-432e-8246-552c6ce6c2de/media__1773201601817.png'
img = cv2.imread(img_path)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold for dark objects
_, binary_dark = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)

# Morphological operations to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(binary_dark, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 400 and area < (w * h * 0.5):
        valid.append(cnt)

print(f"Found {len(valid)} dark objects.")
for c in valid:
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"Box: {x, y, bw, bh}, Area: {cv2.contourArea(c)}")

