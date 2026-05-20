import torch
import cv2
import numpy as np
from src.model1.architecture import PropertyYOLO

class XRayScanner:
    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading Model 1 Stage 2 from {weights_path}...")
        
        # Load our custom ROI-Aligned architecture
        self.model = PropertyYOLO.load_checkpoint(weights_path, device=device).to(device)
        self.model.eval()
        
        self.classes = ['gun', 'knife', 'wrench', 'pliers', 'scissors']
        self.material_map = {0: "Organic", 1: "Fabric/Plastic", 2: "Light Metal", 3: "Heavy/Dense Metal"}

    def scan_image(self, image_path: str):
        # 1. Load and prep the image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x640 (standard YOLO size) while keeping aspect ratio
        img_resized = cv2.resize(img_rgb, (640, 640))
        tensor_img = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            # 2. Stage 1: Get Bounding Boxes using the YOLO backbone
            print("\n🔍 Stage 1: Scanning for Threats...")
            results = self.model.yolo.predict(img_rgb, verbose=False, conf=0.25)[0]
            
            if len(results.boxes) == 0:
                print("✅ Clear: No threats detected.")
                return None

            # Extract boxes into the format ROI-Align expects
            boxes_xyxy = results.boxes.xyxy.to(self.device)
            confidences = results.boxes.conf
            class_ids = results.boxes.cls.int()

            # 3. Stage 2: Extract Physical Properties via ROI-Align
            print(f"⚠️ Detected {len(boxes_xyxy)} potential threats. Running Physics Regression...")
            # Pass the image and the boxes to our custom head
            predictions = self.model.forward_properties(tensor_img, boxes=[boxes_xyxy])
            
            properties = predictions["properties"].cpu().numpy()
            materials = torch.argmax(predictions["material_logits"], dim=1).cpu().numpy() if "material_logits" in predictions else None

            # 4. Format the Output
            self._print_results(class_ids, confidences, properties, materials)

    def _print_results(self, class_ids, confidences, properties, materials):
        print("="*50)
        print(" 🚨 X-RAY THREAT ANALYSIS REPORT ")
        print("="*50)
        
        for i in range(len(class_ids)):
            threat_name = self.classes[class_ids[i]]
            conf = confidences[i] * 100
            prop = properties[i]
            
            print(f"\nTarget {i+1}: {threat_name.upper()} ({conf:.1f}% Match)")
            print("-" * 30)
            
            # Map the 11D array to human-readable physical traits
            print(f"  • Density Level:      {prop[0]:.2f}")
            print(f"  • Edge Sharpness:     {prop[1]:.2f}")
            print(f"  • Symmetry Score:     {prop[2]:.2f}")
            print(f"  • Length/Width Ratio: {prop[3]:.2f}")
            print(f"  • Curvature Index:    {prop[4]:.2f}")
            
            if materials is not None:
                print(f"  • Material Signature: {self.material_map.get(materials[i], 'Unknown')}")
            
            # The all-important Occlusion Score at index 10
            print(f"  • Occlusion Score:    {prop[10]:.2f} (Danger if > 0.8)")

# To run this later:
# scanner = XRayScanner("/content/drive/MyDrive/xray_detection/results/stage2_outputs/stage2_best.pth")
# scanner.scan_image("test_luggage.jpg")