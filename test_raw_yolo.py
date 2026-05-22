from ultralytics import YOLO
import cv2

model = YOLO("checkpoints/best.pt")
img = cv2.imread("runs/detect/train/train_batch0.jpg")
res = model.predict(img)
print(res[0].boxes)
