from ultralytics import YOLO
import cv2

model = YOLO("checkpoints/best.pt")
img = cv2.imread("runs/detect/train/train_batch0.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

res_bgr = model.predict(img, verbose=False)
print("BGR Detections:", len(res_bgr[0].boxes))

res_rgb = model.predict(img_rgb, verbose=False)
print("RGB Detections:", len(res_rgb[0].boxes))
