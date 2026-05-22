import sys
sys.path.append(".")
from backend.routers.inference import api_engine
print("API Engine loaded:", api_engine is not None)
