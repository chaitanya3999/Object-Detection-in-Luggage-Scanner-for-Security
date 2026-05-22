import torch
import traceback
from src.model1.architecture import PropertyYOLO

try:
    print("Initializing PropertyYOLO...")
    model = PropertyYOLO(model_size="yolov8m", num_classes=6, num_properties=11, input_channels=3, pretrained=True, weights_path="checkpoints/best.pt")
    print("Loading stage2_ultimate.pth...")
    checkpoint = torch.load("checkpoints/stage2_ultimate.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("SUCCESSFULLY LOADED BOTH CHECKPOINTS!")
except Exception as e:
    print("FAILED:", e)
    traceback.print_exc()
