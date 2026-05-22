import sys
import os
import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    from main import app
    client_available = True
except ImportError:
    client_available = False

def run_tests():
    """Runs automated integration tests on all FastAPI endpoints."""
    print("=" * 60)
    print("🧪 RUNNING BACKEND INTEGRATION TESTS")
    print("=" * 60)
    
    if not client_available:
        print("⚠ FastAPI TestClient is not available yet. Installing dependencies first.")
        return False
        
    client = TestClient(app)
    
    # 1. Test Health Endpoint
    print("\n🔍 Testing /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200, "Health check failed!"
    print("   Output:", res.json())
    print("✓ Health endpoint OK!")
    
    # 2. Test Model Status
    print("\n🔍 Testing /api/model-status...")
    res = client.get("/api/model-status")
    assert res.status_code == 200, "Model status failed!"
    data = res.json()
    assert "is_dl_mode" in data, "is_dl_mode missing!"
    assert len(data["properties_schema"]) == 11, "Expected 11 dimensions in schema!"
    print(f"   Mode: {'Deep Learning' if data['is_dl_mode'] else 'Computer Vision Fallback'}")
    print("✓ Model status schema OK!")
    
    # 3. Test Statistics
    print("\n🔍 Testing /api/stats...")
    res = client.get("/api/stats")
    assert res.status_code == 200, "Stats failed!"
    data = res.json()
    assert data["total_scanned"] > 0, "Expected scanned count!"
    print(f"   Bags Scanned: {data['total_scanned']}, Bypass Rate: {data['bypass_rate']}%")
    print("✓ Dashboard statistics OK!")
    
    # 4. Test Analytics
    print("\n🔍 Testing /api/mock-analytics...")
    res = client.get("/api/mock-analytics")
    assert res.status_code == 200, "Analytics failed!"
    data = res.json()
    assert "model1_training" in data, "Loss curves missing!"
    assert len(data["feature_importance"]) == 11, "Expected 11 property feature importances!"
    print(f"   Model 1 mAP: {data['metrics']['backbone_mAP50']}, Model 2 F1: {data['metrics']['model2_f1']}")
    print("✓ Performance analytics OK!")
    
    # 5. Test Programmatic Luggage Feeds
    print("\n🔍 Testing programmatic image generator (/api/feed/image/safe_luggage)...")
    res = client.get("/api/feed/image/safe_luggage")
    assert res.status_code == 200, "Image generation failed!"
    data = res.json()
    assert data["image"].startswith("data:image/jpeg;base64,"), "Invalid base64 payload!"
    print("   Output: Image successfully drawn and converted to B64 Data URI!")
    print("✓ Programmatic suitcase generator OK!")
    
    print("\n" + "=" * 60)
    print("🎉 ALL ENDPOINT INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_tests()
