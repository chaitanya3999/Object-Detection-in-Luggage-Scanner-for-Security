import sys
import os
import traceback

try:
    from backend.api import XRayAPI
    root_dir = os.path.abspath(".")
    stage1_path = os.path.join(root_dir, "checkpoints", "best.pt")
    stage2_path = os.path.join(root_dir, "checkpoints", "stage2_ultimate.pth")
    api = XRayAPI(stage1_weights=stage1_path, stage2_weights=stage2_path, device="cpu")
    print("SUCCESS")
except Exception as e:
    print("FAILED")
    traceback.print_exc()
