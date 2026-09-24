"""
Prescriptive Maintenance Recommendation Engine for CNC Manufacturing.
Translates tool health, predicted flank wear, remaining useful life (RUL),
and anomaly alarms into actionable shop-floor decisions, SOP checklists,
and downtime risk assessments.
"""

from typing import Dict, Any, List
from utils.config import (
    WEAR_WARNING_UM,
    WEAR_CRITICAL_UM,
    WEAR_MAX_THRESHOLD_UM,
    STATUS_NORMAL,
    STATUS_WARNING,
    STATUS_CRITICAL
)


def get_maintenance_decision(
    tool_wear_um: float,
    tool_health_pct: float,
    rul_cuts: int,
    anomaly_status: str = STATUS_NORMAL,
    anomaly_reason: str = ""
) -> Dict[str, Any]:
    """
    Generate prescriptive maintenance recommendations according to
    the 4-tier decision architecture:
    1. Normal: "Continue Operation"
    2. Warning: "Inspect Tool"
    3. High Wear: "Schedule Maintenance"
    4. Critical: "Replace Tool"
    """
    wear = float(tool_wear_um)
    is_critical_anomaly = (anomaly_status == STATUS_CRITICAL)
    is_warning_anomaly = (anomaly_status == STATUS_WARNING)

    if wear >= WEAR_MAX_THRESHOLD_UM or rul_cuts <= 3 or is_critical_anomaly:
        rec_title = "Replace Tool"
        badge_level = "CRITICAL"
        color = "#EF4444"
        urgency = "IMMEDIATE ACTION REQUIRED"
        summary = (
            f"Flank wear ({wear:.1f} µm) has breached the critical threshold ({WEAR_MAX_THRESHOLD_UM:.0f} µm) "
            f"or a severe anomaly was detected ({anomaly_reason}). Continued cutting risks catastrophic "
            "tool fracture, workpiece damage, and spindle overload."
        )
        checklist = [
            "Trigger Program Feed-Hold at completion of current pass (or E-Stop if chatter is active).",
            "Retract spindle Z-axis and switch off high-pressure coolant supply.",
            "Lockout/Tagout (LOTO) spindle motor before manual intervention.",
            "Remove worn 4-flute carbide end-mill and inspect toolholder collet for fretting.",
            "Clean HSK/BT spindle taper socket and check for debris or chip welding.",
            "Install fresh cutter, torque collet nut to specified Newton-meters.",
            "Run laser/optical tool setter cycle to re-calibrate tool length and radius offsets.",
            "Update tool life management register in CNC controller."
        ]
        estimated_downtime_min = 15
        scrap_risk = "HIGH - Catastrophic Part Ruin Risk"

    elif wear >= WEAR_CRITICAL_UM or rul_cuts <= 15:
        rec_title = "Schedule Maintenance"
        badge_level = "HIGH WEAR"
        color = "#F97316"  # Orange
        urgency = "SCHEDULE WITHIN CURRENT SHIFT"
        summary = (
            f"Tool flank wear ({wear:.1f} µm) is entering accelerated tertiary wear with only ~{rul_cuts} cuts "
            "remaining. Surface finish roughness is deteriorating. Arrange tool replacement at next scheduled part transfer."
        )
        checklist = [
            "Alert tool crib supervisor to stage replacement cutter and verified toolholder.",
            "Inspect machined part surface finish (Ra test) for micro-chatter marks or burring.",
            "Verify coolant concentration (refractometer target: 8.5% - 10.5%).",
            "Prepare CNC controller tool offset offsets for upcoming tool exchange.",
            "Plan tool swap during the upcoming batch unload cycle to minimize non-productive downtime."
        ]
        estimated_downtime_min = 12
        scrap_risk = "MODERATE - Surface finish nearing out-of-tolerance reject"

    elif wear >= WEAR_WARNING_UM or is_warning_anomaly or rul_cuts <= 40:
        rec_title = "Inspect Tool"
        badge_level = "WARNING"
        color = "#F59E0B"  # Yellow/Amber
        urgency = "INSPECTION ADVISORY"
        summary = (
            f"Tool wear ({wear:.1f} µm) has exceeded nominal break-in limits or anomaly detection flagged an "
            f"irregular vibration/force signature ({anomaly_reason}). Perform visual optical inspection."
        )
        checklist = [
            "Perform in-spindle visual inspection of cutting flutes for edge chipping or built-up edge (BUE).",
            "Check coolant nozzle alignment to ensure high-pressure jet directly hits the tool-workpiece interface.",
            "Verify workpiece clamping rigidity and pneumatic vise pressure.",
            "Review acoustic emission and vibration trends over the last 10 milling passes."
        ]
        estimated_downtime_min = 5
        scrap_risk = "LOW - Part within dimensional tolerance"

    else:
        rec_title = "Continue Operation"
        badge_level = "NORMAL"
        color = "#10B981"  # Green
        urgency = "NOMINAL PRODUCTION"
        summary = (
            f"Tool condition is healthy ({tool_health_pct:.1f}% health index, {wear:.1f} µm wear). "
            f"All forces, vibration, and acoustic signals are well within normal operating envelopes."
        )
        checklist = [
            "Maintain continuous automated milling cycle.",
            "Monitor live force and vibration telemetry for unexpected drift.",
            "Verify routine chip conveyor operation and swarf clearance."
        ]
        estimated_downtime_min = 0
        scrap_risk = "NEGLIGIBLE - High precision machining"

    return {
        "recommendation": rec_title,
        "level": badge_level,
        "color": color,
        "urgency": urgency,
        "summary": summary,
        "checklist": checklist,
        "estimated_downtime_min": estimated_downtime_min,
        "scrap_risk": scrap_risk,
        "tool_health_pct": round(tool_health_pct, 1),
        "tool_wear_um": round(wear, 1),
        "rul_cuts": rul_cuts
    }
