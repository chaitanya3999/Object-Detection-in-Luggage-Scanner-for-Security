import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db, ScanLog

router = APIRouter(prefix="/api")

@router.get("/mock-analytics")
def get_mock_analytics():
    """Returns analytics data for training performance, model accuracy, and explainable feature importances."""
    # Model 1 Loss Curves
    epochs = list(range(1, 61))
    # Loss curves showing standard 2-stage training
    # Stage 1: Detection train (1 to 30)
    # Stage 2: Property head train (31 to 60)
    stage1_loss = [4.2 * (0.91**e) + 0.5 for e in range(1, 31)]
    stage2_loss = [2.8 * (0.94**e) + 0.3 for e in range(1, 31)]
    mAP_curve = [0.1 + 0.72 * (1.0 - 0.85**e) for e in range(1, 31)] + [0.82] * 30
    
    # Combined training curves
    stage1_curve = [{"epoch": e, "val_loss": stage1_loss[e-1], "mAP": mAP_curve[e-1], "stage": 1} for e in range(1, 31)]
    # Stage 2 losses focus on Property MAE/RMSE
    stage2_curve = [{"epoch": e + 30, "property_mae": stage2_loss[e-1], "mAP": 0.82, "stage": 2} for e in range(1, 31)]
    
    # Model 2 Property Importance Score (RF/XGBoost)
    feature_importance = [
        {"property": "Density Level", "importance": 0.28, "explanation": "Dark/absorbed pixels represent shielding or thick metals"},
        {"property": "Sharp Edge Count", "importance": 0.19, "explanation": "High edge counts distinguish weapons/tools from organic shapes"},
        {"property": "Material Category", "importance": 0.16, "explanation": "Metals (Blue) and Opaque (Black) pose structural risks"},
        {"property": "Edge Sharpness", "importance": 0.11, "explanation": "High Canny gradient density marks sharp objects"},
        {"property": "Length-to-Width Ratio", "importance": 0.08, "explanation": "Aspect ratio isolates long wires or metal rods"},
        {"property": "Avg Absorption Intensity", "importance": 0.06, "explanation": "Differentiates organic powders from dense liquids"},
        {"property": "Approximate Volume", "importance": 0.04, "explanation": "Assesses bulk scale of containers"},
        {"property": "Occlusion Score", "importance": 0.03, "explanation": "Tracks items masked behind main structures"},
        {"property": "Curvature Index", "importance": 0.02, "explanation": "Indicates irregular contour profiles"},
        {"property": "Symmetry Score", "importance": 0.02, "explanation": "Identifies asymmetric threat objects"},
        {"property": "Material Homogeneity", "importance": 0.01, "explanation": "Homogeneous elements track plastics vs complex items"}
    ]
    
    # Confusion Matrix (Model 2 classes: Safe vs Threat)
    confusion_matrix = {
        "labels": ["Safe", "Threat"],
        "matrix": [
            [942, 18],  # True Safe, False Threat (False alarm)
            [12, 384]   # False Safe (Miss), True Threat
        ]
    }
    
    return {
        "model1_training": stage1_curve + stage2_curve,
        "feature_importance": feature_importance,
        "confusion_matrix": confusion_matrix,
        "metrics": {
            "backbone_mAP50": 0.824,
            "property_reg_mae": 0.042,
            "material_accuracy": 0.912,
            "model2_precision": 0.955,
            "model2_recall": 0.969,
            "model2_f1": 0.962
        }
    }

@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Returns general luggage throughput and detection statistics based on DB + mocked base."""
    # Get stats from DB
    total_db_scans = db.query(ScanLog).count()
    threats_in_db = db.query(ScanLog).filter(ScanLog.overall_threat.in_(["WARNING", "CRITICAL"])).count()
    
    # Combine with mock data for a realistic baseline
    base_scans = 1284
    base_threats = 47
    
    return {
        "total_scanned": base_scans + total_db_scans,
        "threats_detected": base_threats + threats_in_db,
        "false_alarms": 18,
        "bypass_rate": round(100 - ((base_threats + threats_in_db) / max(1, (base_scans + total_db_scans)) * 100), 1),
        "material_distribution": [
            {"name": "Organic (Orange)", "value": 68.2},
            {"name": "Metallic (Blue)", "value": 18.5},
            {"name": "Mixed (Green)", "value": 11.1},
            {"name": "Opaque (Black)", "value": 2.2}
        ],
        "operator_stats": {
            "avg_inspection_time_sec": 4.6,
            "throughput_bags_per_min": 14.2,
            "missed_threats": 0
        }
    }

@router.get("/scan-logs")
def get_recent_scans(limit: int = 20, db: Session = Depends(get_db)):
    """Fetch recent scan history for the Alert Log sidebar."""
    logs = db.query(ScanLog).order_by(ScanLog.timestamp.desc()).limit(limit).all()
    result = []
    for log in logs:
        try:
            threats = json.loads(log.threat_details)
        except:
            threats = []
        result.append({
            "id": log.id,
            "timestamp": log.timestamp.isoformat() + "Z",
            "overall_threat": log.overall_threat,
            "num_objects": log.num_objects,
            "threats": threats,
            "inference_mode": log.inference_mode
        })
    return result
