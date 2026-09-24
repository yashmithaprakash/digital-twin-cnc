"""
Headless page logic verification to ensure zero exceptions across all 9 pages.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from utils.config import (
    LIMITS, COLORS, STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL,
    WEAR_WARNING_UM, WEAR_CRITICAL_UM, WEAR_MAX_THRESHOLD_UM,
    NOMINAL_SPINDLE_SPEED_RPM, NOMINAL_FEED_RATE_MM_MIN,
    NOMINAL_DEPTH_OF_CUT_MM, NOMINAL_TOOL_DIAMETER_MM,
    NUM_FLUTES, MATERIALS
)
from src.data_loader import load_dataset, generate_cut_waveform, init_db, get_maintenance_history
from src.tool_wear_model import get_trained_tool_wear_model
from src.anomaly_detection import get_trained_anomaly_detector
from src.rul_prediction import RULEstimator
from src.simulation import run_what_if_simulation
from src.recommendations import get_maintenance_decision

def test_all_pages():
    print("Testing all page logic headlessly...")
    df, is_real, source_label = load_dataset()
    wear_model = get_trained_tool_wear_model(df)
    anomaly_detector = get_trained_anomaly_detector(df)

    cut_record = df.iloc[74].to_dict()
    eval_res = anomaly_detector.evaluate_record(cut_record)
    pred_wear_arr = wear_model.predict(pd.DataFrame([cut_record]))
    predicted_wear = float(pred_wear_arr[0])
    tool_health = wear_model.predict_health_pct(predicted_wear)
    rul_engine = RULEstimator(wear_limit_um=WEAR_MAX_THRESHOLD_UM)
    rul_result = rul_engine.estimate_rul(predicted_wear, df.iloc[:75]["flank_wear_um"].tolist())
    maint_rec = get_maintenance_decision(predicted_wear, tool_health, rul_result["rul_cuts"], eval_res["status"], eval_res["reason"])

    print("  [1/9] Dashboard Home Logic... OK")
    
    # 2. Sensor Telemetry
    wf = generate_cut_waveform(75, predicted_wear, num_points=400)
    assert len(wf) > 0
    print("  [2/9] Sensor Telemetry Logic... OK")

    # 3. Tool Health
    assert 0 <= tool_health <= 100
    assert len(wear_model.feature_importances_) > 0
    print("  [3/9] Tool Health Logic... OK")

    # 4. Anomaly Detection
    preds, scores = anomaly_detector.score_samples(df.iloc[:20])
    assert len(scores) == 20
    print("  [4/9] Anomaly Detection Logic... OK")

    # 5. RUL Prediction
    assert len(rul_result["trajectory_wear"]) > 0
    print("  [5/9] RUL Prediction Logic... OK")

    # 6. Digital Twin
    from src.digital_twin_view import generate_cnc_machine_svg
    sp_rpm = float(cut_record.get("spindle_speed_rpm", NOMINAL_SPINDLE_SPEED_RPM))
    vc = (np.pi * NOMINAL_TOOL_DIAMETER_MM * sp_rpm) / 1000.0
    fc = float(cut_record.get("force_resultant", 150))
    power_kw = (fc * vc) / 60000.0
    assert power_kw > 0
    # Test SVG generator across Normal, Warning, and Critical states
    for state in ["NORMAL", "WARNING", "CRITICAL"]:
        svg = generate_cnc_machine_svg(
            cut_id=75,
            spindle_speed_rpm=sp_rpm,
            feed_rate_mm_min=1555.0,
            depth_of_cut_mm=1.5,
            coolant_mode="Flood",
            flank_wear_um=predicted_wear,
            tool_health_pct=tool_health,
            machine_status=state,
            force_res_n=fc,
            vibration_rms_g=1.2
        )
        assert "<svg" in svg and "</svg>" in svg
    print("  [6/9] Digital Twin Virtual Machine SVG Logic... OK")

    # 7. What-if Simulation
    sim = run_what_if_simulation(10400, 1555, 1.5, "Flood", "Inconel 718", predicted_wear)
    assert sim["wear_acceleration_factor"] > 0
    print("  [7/9] What-if Simulation Logic... OK")

    # 8. Maintenance Advisory
    assert len(maint_rec["checklist"]) > 0
    print("  [8/9] Maintenance Advisory Logic... OK")

    # 9. Dataset Management
    assert len(df.describe()) > 0
    print("  [9/9] Dataset Management Logic... OK")

    print("\nALL 9 PAGES TESTED AND VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all_pages()
