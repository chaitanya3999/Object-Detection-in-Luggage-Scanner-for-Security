from ultralytics import YOLO
import ultralytics
import yaml
from pathlib import Path

yaml_path = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8.yaml"
custom_yaml_path = "test_custom.yaml"
with open(yaml_path, "r") as f: d = yaml.safe_load(f)
d["ch"] = 3
d["nc"] = 6
with open(custom_yaml_path, "w") as f: yaml.dump(d, f)

model = YOLO(custom_yaml_path)
try:
    model.load("checkpoints/best.pt")
    print("LOADED BEST.PT SUCCESSFULLY")
except Exception as e:
    print("FAILED TO LOAD BEST.PT:", e)
