"""
Verification test script for all CNC Digital Twin components.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import (
    load_dataset, init_db, generate_cut_waveform,
    log_telemetry, log_maintenance_event, get_maintenance_history
)
from src.feature_engineering import extract_cut_level_features
from src.tool_wear_model import get_trained_tool_wear_model
from src.anomaly_detection import get_trained_anomaly_detector
from src.rul_prediction import RULEstimator
from src.simulation import run_what_if_simulation
from src.recommendations import get_maintenance_decision

def run_tests():
    print("Testing Data Loader...")
    df, is_real, lbl = load_dataset()
    assert len(df) > 0, "DataFrame empty"
    print(f"  [OK] Loaded dataset with {len(df)} cuts. Source: {lbl}")

    print("Testing Waveform Generator...")
    wf = generate_cut_waveform(cut_id=1, flank_wear_um=30.0)
    assert len(wf) > 0, "Waveform empty"
    print(f"  [OK] Waveform generated with {len(wf)} points.")

    print("Testing Tool Wear Model...")
    model = get_trained_tool_wear_model(df)
    pred = model.predict(df.iloc[:5])
    assert len(pred) == 5, "Prediction length mismatch"
    health = model.predict_health_pct(pred[0])
    assert 0 <= health <= 100, "Health score range error"
    print(f"  [OK] Wear Model Test Passed. Pred: {pred[0]:.2f} um, Health: {health:.1f}%")

    print("Testing Anomaly Detector...")
    ad = get_trained_anomaly_detector(df)
    eval_res = ad.evaluate_record(df.iloc[0].to_dict())
    assert "status" in eval_res, "Status missing"
    print(f"  [OK] Anomaly Detector Test Passed: {eval_res['status']}, score: {eval_res['anomaly_score']}")

    print("Testing RUL Estimator...")
    rul_engine = RULEstimator()
    rul_res = rul_engine.estimate_rul(current_wear_um=80.0, wear_history=[70, 72, 75, 78, 80])
    assert rul_res["rul_cuts"] > 0, "RUL cuts must be > 0"
    print(f"  [OK] RUL Test Passed: {rul_res['rul_cuts']} cuts, {rul_res['rul_minutes']} mins")

    print("Testing Simulation...")
    sim_res = run_what_if_simulation(12000, 1800, 2.0, "Mist (MQL)", "Titanium Ti-6Al-4V", 50.0)
    assert "simulated_rul_cuts" in sim_res, "Simulated RUL missing"
    print(f"  [OK] Simulation Test Passed: {sim_res['machine_condition']}, acceleration: {sim_res['wear_acceleration_factor']}x")

    print("Testing Maintenance Decision...")
    rec = get_maintenance_decision(170.0, 20.0, 5, "WARNING", "High vibration")
    assert rec["recommendation"] in ["Schedule Maintenance", "Replace Tool"]
    print(f"  [OK] Recommendation Test Passed: {rec['recommendation']}")

    print("Testing Database Logging...")
    init_db()
    log_maintenance_event("Inspection", "Tool-1", 45.0, 50, "Tech-1", "Checked flutes", "All good")
    hist = get_maintenance_history()
    assert len(hist) > 0, "History empty"
    print(f"  [OK] DB Test Passed. {len(hist)} maintenance records.")

    print("\n>>> ALL CORE MODULE TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_tests()
