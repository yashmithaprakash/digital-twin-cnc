"""
What-If Manufacturing Simulation Engine for CNC Milling.
Evaluates the impact of varying machining parameters (Spindle Speed, Feed Rate,
Depth of Cut, Coolant Mode, and Workpiece Material) on cutting forces, vibration,
tool wear acceleration rate, RUL, and operational safety.
"""

import numpy as np
from typing import Dict, Any

from utils.config import (
    NOMINAL_SPINDLE_SPEED_RPM,
    NOMINAL_FEED_RATE_MM_MIN,
    NOMINAL_DEPTH_OF_CUT_MM,
    NOMINAL_TOOL_DIAMETER_MM,
    NUM_FLUTES,
    WEAR_MAX_THRESHOLD_UM,
    MATERIALS,
    STATUS_NORMAL,
    STATUS_WARNING,
    STATUS_CRITICAL
)


def run_what_if_simulation(
    spindle_speed_rpm: float,
    feed_rate_mm_min: float,
    depth_of_cut_mm: float,
    coolant_mode: str = "Flood",
    material_name: str = "Inconel 718",
    current_wear_um: float = 60.0
) -> Dict[str, Any]:
    """
    Simulate machining performance and wear acceleration using physics-informed
    Taylor tool life and Kienzle cutting mechanics approximations.
    """
    mat_props = MATERIALS.get(material_name, MATERIALS["Inconel 718"])

    # 1. Machining Kinematics
    tool_diameter_mm = NOMINAL_TOOL_DIAMETER_MM
    cutting_speed_vc = (np.pi * tool_diameter_mm * spindle_speed_rpm) / 1000.0  # m/min
    feed_per_tooth_fz = feed_rate_mm_min / (spindle_speed_rpm * NUM_FLUTES)    # mm/tooth
    nominal_vc = (np.pi * tool_diameter_mm * NOMINAL_SPINDLE_SPEED_RPM) / 1000.0
    nominal_fz = NOMINAL_FEED_RATE_MM_MIN / (NOMINAL_SPINDLE_SPEED_RPM * NUM_FLUTES)

    # 2. Material Removal Rate (MRR in cm³/min)
    radial_depth_mm = tool_diameter_mm * 0.75  # 75% immersion
    mrr_cm3_min = (depth_of_cut_mm * radial_depth_mm * feed_rate_mm_min) / 1000.0

    # 3. Coolant Effect Modifiers
    coolant_wear_factors = {
        "Flood": 1.0,
        "Mist (MQL)": 1.35,
        "Dry / Air Blast": 2.25
    }
    coolant_force_factors = {
        "Flood": 1.0,
        "Mist (MQL)": 1.12,
        "Dry / Air Blast": 1.28
    }
    c_wear_mult = coolant_wear_factors.get(coolant_mode, 1.0)
    c_force_mult = coolant_force_factors.get(coolant_mode, 1.0)

    # 4. Cutting Force Estimation (Kienzle Model approximation)
    kc1 = mat_props["specific_cutting_force_kc1"]
    force_ratio_speed = (cutting_speed_vc / nominal_vc) ** (-0.12)
    force_ratio_feed = (feed_per_tooth_fz / nominal_fz) ** 0.75
    force_ratio_doc = (depth_of_cut_mm / NOMINAL_DEPTH_OF_CUT_MM) ** 0.95
    wear_force_factor = 1.0 + (current_wear_um / 220.0)

    est_force_res = (
        160.0
        * force_ratio_feed
        * force_ratio_doc
        * force_ratio_speed
        * c_force_mult
        * wear_force_factor
        * (mat_props["machinability_index"] / 0.35) ** (-0.4)
    )

    # 5. Vibration Estimation (Chatter risk increases with high RPM and DOC)
    speed_ratio = spindle_speed_rpm / NOMINAL_SPINDLE_SPEED_RPM
    est_vibration_rms = (
        1.05
        * (speed_ratio ** 1.1)
        * (force_ratio_doc ** 0.6)
        * (c_wear_mult ** 0.3)
        * wear_force_factor
    )

    # 6. Acoustic Emission (sensitive to high shear & friction)
    est_ae_rms = 0.22 * (speed_ratio ** 0.8) * (c_wear_mult ** 0.5) * wear_force_factor

    # 7. Taylor Tool Life Acceleration Factor
    # Extended Taylor equation: T = C / (V^(1/n) * f^(a/n) * ap^(b/n))
    n = mat_props["taylor_n"]
    a = mat_props["feed_exponent_a"]
    b = mat_props["doc_exponent_b"]

    speed_factor = (cutting_speed_vc / nominal_vc) ** (1.0 / n)
    feed_factor = (feed_per_tooth_fz / nominal_fz) ** (a / n)
    doc_factor = (depth_of_cut_mm / NOMINAL_DEPTH_OF_CUT_MM) ** (b / n)

    wear_acceleration = float(speed_factor * feed_factor * doc_factor * c_wear_mult)
    wear_acceleration = float(np.clip(wear_acceleration, 0.2, 8.0))

    # Base wear rate in µm/cut at nominal condition
    base_wear_rate = 0.55
    simulated_wear_rate = base_wear_rate * wear_acceleration

    # Estimated RUL under simulated parameters
    wear_headroom = max(0.0, WEAR_MAX_THRESHOLD_UM - current_wear_um)
    simulated_rul_cuts = int(np.round(wear_headroom / max(simulated_wear_rate, 0.1)))
    nominal_rul_cuts = int(np.round(wear_headroom / base_wear_rate))
    rul_delta_pct = round(((simulated_rul_cuts - nominal_rul_cuts) / max(1, nominal_rul_cuts)) * 100.0, 1)

    # Surface Roughness Ra proxy (theoretical kinematic roughness: Ra ~ fz² / (32 * r_tool))
    theoretical_ra_um = round((feed_per_tooth_fz ** 2) / (32.0 * 0.8) * 1000.0 + (current_wear_um / 80.0), 2)

    # 8. Machine Condition Classification & Recommendation
    if est_force_res > 380.0 or est_vibration_rms > 3.6 or wear_acceleration > 3.2:
        condition = STATUS_CRITICAL
        rec = "Parameters exceed safe machine thresholds. High risk of tool breakage, spindle overload, or chatter mark rejects. Reduce feed or depth of cut immediately."
    elif est_force_res > 280.0 or est_vibration_rms > 2.2 or wear_acceleration > 1.6:
        condition = STATUS_WARNING
        rec = "Aggressive machining regime. Accelerated tool degradation will occur. Increase coolant flow or moderate feed rate to prolong cutter life."
    else:
        condition = STATUS_NORMAL
        rec = "Optimal machining regime. Stable cutting dynamics, well-controlled tool wear rate, and good surface integrity expected."

    # Trajectory extrapolation for comparison plotting
    traj_steps = min(60, max(15, simulated_rul_cuts + 10))
    traj_cuts = np.arange(0, traj_steps + 1)
    traj_wear = []
    for step in traj_cuts:
        w = current_wear_um + simulated_wear_rate * step * (1.0 + 0.012 * step if current_wear_um > 120 else 1.0)
        traj_wear.append(round(float(min(WEAR_MAX_THRESHOLD_UM + 25.0, w)), 2))

    # Machining time per pass
    time_per_pass = (100.0 / max(100.0, feed_rate_mm_min)) + 0.08
    simulated_rul_minutes = round(simulated_rul_cuts * time_per_pass, 1)

    # Anomaly risk level
    if condition == STATUS_CRITICAL:
        anomaly_risk = "HIGH (ANOMALY)"
    elif condition == STATUS_WARNING:
        anomaly_risk = "MODERATE (WARNING)"
    else:
        anomaly_risk = "LOW (NORMAL)"

    expected_health_pct = round(max(0.0, 100.0 * (1.0 - (current_wear_um / WEAR_MAX_THRESHOLD_UM))), 1)
    projected_wear_10_cuts = round(min(WEAR_MAX_THRESHOLD_UM + 15.0, current_wear_um + simulated_wear_rate * 10), 1)

    disclaimer = (
        "Data-driven simulation estimate: This simulation couples empirical Taylor tool life equations "
        "and Kienzle force modeling. Values serve for comparative process optimization and "
        "what-if exploration. Do not present or rely on it as a physically validated CNC simulation."
    )

    return {
        "cutting_speed_vc_m_min": round(cutting_speed_vc, 1),
        "feed_per_tooth_fz_mm": round(feed_per_tooth_fz, 4),
        "mrr_cm3_min": round(mrr_cm3_min, 2),
        "estimated_force_res_n": round(est_force_res, 1),
        "estimated_vibration_rms_g": round(est_vibration_rms, 2),
        "estimated_ae_rms_v": round(est_ae_rms, 3),
        "wear_acceleration_factor": round(wear_acceleration, 2),
        "simulated_wear_rate_um_cut": round(simulated_wear_rate, 3),
        "simulated_rul_cuts": simulated_rul_cuts,
        "simulated_rul_minutes": simulated_rul_minutes,
        "nominal_rul_cuts": nominal_rul_cuts,
        "rul_delta_pct": rul_delta_pct,
        "surface_roughness_ra_um": theoretical_ra_um,
        "machine_condition": condition,
        "anomaly_risk": anomaly_risk,
        "expected_tool_health_pct": expected_health_pct,
        "projected_wear_10_cuts": projected_wear_10_cuts,
        "maintenance_recommendation": rec,
        "trajectory_cuts": traj_cuts.tolist(),
        "trajectory_wear": traj_wear,
        "disclaimer": disclaimer,
        "material": material_name,
        "coolant": coolant_mode
    }
