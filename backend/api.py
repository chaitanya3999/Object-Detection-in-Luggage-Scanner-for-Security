import cv2
import json
import torch
import os
from src.model1.architecture import PropertyYOLO

class XRayAPI:
    def __init__(self, stage1_weights: str, stage2_weights: str, device: str = "cpu"):
        self.device = device
        print(f"🔧 Spinning up X-Ray API Engine on {device.upper()}...")
        
        # Initialize Architecture
        self.model = PropertyYOLO(
            model_size="yolov8m",
            num_classes=5,
            num_properties=11,
            input_channels=3,
            pretrained=True,
            weights_path=stage1_weights
        ).to(device)
        
        # Inject Physics Knowledge
        checkpoint = torch.load(stage2_weights, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Filter out shape mismatches and ALL YOLO weights (to perfectly preserve best.pt)
        current_state = self.model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if k in current_state and current_state[k].shape == v.shape and not k.startswith("yolo.")
        }
        
        self.model.load_state_dict(filtered_state_dict, strict=False)
            
        self.model.eval()

    def scan_luggage(self, image_path: str) -> str:
        """Scans an image and returns a highly detailed JSON payload for the UI."""
        img = cv2.imread(image_path)
        if img is None:
            return json.dumps({"error": "Image not found or unreadable."})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        tensor_img = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        payload = {
            "status": "clear",
            "security_protocol": "STANDARD_CLEARANCE",
            "threat_count": 0,
            "diagnostics": {
                "background_noise_events": 0,
                "composition_breakdown": {
                    "Heavy/Dense Metal": "0%",
                    "Light Metal": "0%",
                    "Fabric/Plastic": "0%",
                    "Organic": "0%"
                }
            },
            "detections": []
        }

        with torch.no_grad():
            results = self.model.yolo.predict(img, verbose=False, conf=0.25)[0]
            
            if len(results.boxes) == 0:
                return json.dumps(payload, indent=4)

            boxes_xyxy = results.boxes.xyxy.to(self.device)
            confidences = results.boxes.conf
            class_ids = results.boxes.cls.int()

            predictions = self.model.forward_properties(tensor_img, boxes=[boxes_xyxy])
            properties = predictions["properties"].cpu().numpy()
            
            class_names = ['gun', 'knife', 'wrench', 'pliers', 'scissors']
            valid_detections = []
            
            noise_events = 0
            material_counts = {"Heavy/Dense Metal": 0, "Light Metal": 0, "Fabric/Plastic": 0, "Organic": 0}
            highest_severity = 0
            gun_detected = False

            for i in range(len(class_ids)):
                class_id_int = int(class_ids[i].item())
                
                # Filter out background artifact
                if class_id_int == 5:
                    noise_events += 1
                    continue
                
                threat_name = class_names[class_id_int] if 0 <= class_id_int < len(class_names) else f"UNKNOWN_{class_id_int}"
                if threat_name == 'gun':
                    gun_detected = True
                
                prop = properties[i]
                
                predicted_density = float(prop[0])
                if predicted_density > 0.75:
                    material_label = "Heavy/Dense Metal"
                elif predicted_density > 0.45:
                    material_label = "Light Metal"
                elif predicted_density > 0.20:
                    material_label = "Fabric/Plastic"
                else:
                    material_label = "Organic"
                    
                material_counts[material_label] += 1

                conf_val = float(confidences[i])
                severity_score = int(min(100, round((conf_val * 0.4 + predicted_density * 0.6) * 100)))
                if severity_score > highest_severity:
                    highest_severity = severity_score

                detection = {
                    "id": len(valid_detections) + 1,
                    "classification": threat_name.upper(),
                    "threat_severity_index": severity_score,
                    "confidence_score": round(conf_val, 3),
                    "bounding_box": [round(float(x), 1) for x in boxes_xyxy[i]],
                    "physical_properties": {
                        "density_level": round(predicted_density, 3),
                        "edge_sharpness": round(float(prop[1]), 3),
                        "symmetry_score": round(float(prop[2]), 3),
                        "length_width_ratio": round(float(prop[3]), 3),
                        "curvature_index": round(float(prop[4]), 3),
                        "occlusion_score": round(float(prop[10]), 3)
                    },
                    "material_signature": material_label
                }
                valid_detections.append(detection)

            total_items = sum(material_counts.values())
            if total_items > 0:
                for mat in material_counts:
                    percentage = int(round((material_counts[mat] / total_items) * 100))
                    payload["diagnostics"]["composition_breakdown"][mat] = f"{percentage}%"
            
            payload["diagnostics"]["background_noise_events"] = noise_events

            if len(valid_detections) == 0:
                payload["status"] = "clear"
                payload["security_protocol"] = "STANDARD_CLEARANCE"
                payload["threat_count"] = 0
            else:
                payload["status"] = "threat_detected"
                payload["threat_count"] = len(valid_detections)
                payload["detections"] = valid_detections
                
                if gun_detected or highest_severity >= 85:
                    payload["security_protocol"] = "CRITICAL_LOCKDOWN_INITIATED"
                elif highest_severity >= 50:
                    payload["security_protocol"] = "MANUAL_INSPECTION_REQUIRED"
                else:
                    payload["security_protocol"] = "CAUTION_FLAGGED"

        return json.dumps(payload, indent=4)