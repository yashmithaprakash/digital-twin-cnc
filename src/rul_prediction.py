"""
Remaining Useful Life (RUL) Estimation Engine for CNC Milling.
Transparent, explainable wear trajectory extrapolation based on linear-exponential
degradation models, wear velocity slopes, and ISO 8688 tool wear criteria.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.config import WEAR_MAX_THRESHOLD_UM, WEAR_WARNING_UM, NOMINAL_FEED_RATE_MM_MIN


class RULEstimator:
    """
    Explainable RUL prediction system combining degradation regression with
    empirical tool life degradation slopes.
    """

    def __init__(self, wear_limit_um: float = WEAR_MAX_THRESHOLD_UM):
        self.wear_limit_um = wear_limit_um

    def estimate_rul(self, current_wear_um: float,
                     wear_history: Optional[List[float]] = None,
                     feed_rate_mm_min: float = NOMINAL_FEED_RATE_MM_MIN,
                     cut_length_mm: float = 100.0) -> Dict[str, Any]:
        """
        Calculate explainable RUL in remaining cuts, machining minutes, and bounds.
        
        Args:
            current_wear_um: Current or ML-predicted flank wear VB (µm).
            wear_history: Recent sequence of wear measurements to compute wear rate.
            feed_rate_mm_min: Table feed rate in mm/min.
            cut_length_mm: Physical length of each milling pass (default 100 mm).
            
        Returns:
            Dictionary containing RUL predictions, trajectory forecasts, and transparent assumptions.
        """
        current_wear = max(5.0, float(current_wear_um))
        wear_headroom = max(0.0, self.wear_limit_um - current_wear)

        # 1. Determine local wear degradation rate (µm per cut)
        if wear_history is not None and len(wear_history) >= 3:
            recent_wear = np.array(wear_history[-15:], dtype=float)
            x = np.arange(len(recent_wear))
            # Linear slope of recent wear points
            slope, _ = np.polyfit(x, recent_wear, 1)
            wear_rate = max(0.25, float(slope))
        else:
            # Physics-based baseline degradation rate:
            # - Steady-state wear rate is typically ~0.4 - 0.7 µm/cut for Inconel milling
            # - Accelerated tertiary wear rate jumps to ~1.2 - 2.5 µm/cut when VB > 140µm
            if current_wear < 60.0:
                wear_rate = 0.45
            elif current_wear < WEAR_WARNING_UM:
                wear_rate = 0.58
            else:
                # Exponential acceleration in tertiary wear stage
                acceleration_factor = 1.0 + ((current_wear - WEAR_WARNING_UM) / 40.0) ** 1.8
                wear_rate = min(3.5, 0.65 * acceleration_factor)

        # 2. Tool Degradation Phase classification
        if current_wear < 45.0:
            phase = "Stage 1: Initial Break-in Wear"
            desc = "Edge rounding and micro-relief stabilization. Low risk of tool failure."
        elif current_wear < WEAR_WARNING_UM:
            phase = "Stage 2: Steady-State Uniform Wear"
            desc = "Stable linear flank abrasion. Predictable degradation rate."
        else:
            phase = "Stage 3: Accelerated Tertiary Wear"
            desc = "Severe thermal softening and micro-chipping. Accelerated failure risk."

        # 3. RUL in cuts
        if wear_headroom <= 0:
            rul_cuts = 0
            rul_lower = 0
            rul_upper = 0
        else:
            rul_cuts = int(np.round(wear_headroom / wear_rate))
            # Uncertainty bounds (±18% based on stochastic cutting variation)
            rul_lower = int(max(0, np.floor(rul_cuts * 0.82)))
            rul_upper = int(np.ceil(rul_cuts * 1.18))

        # 4. Machining time calculation
        # Time per cut (minutes) = cut_length / feed_rate + rapid return time (approx 0.1 min)
        time_per_cut_min = (cut_length_mm / max(100.0, feed_rate_mm_min)) + 0.08
        rul_minutes = round(rul_cuts * time_per_cut_min, 1)
        rul_minutes_lower = round(rul_lower * time_per_cut_min, 1)
        rul_minutes_upper = round(rul_upper * time_per_cut_min, 1)

        # 5. Generate Future Degradation Trajectory (for Plotly curve)
        proj_steps = min(60, max(15, rul_cuts + 10))
        proj_cuts = np.arange(0, proj_steps + 1)
        proj_wear = []
        for step in proj_cuts:
            if current_wear >= WEAR_WARNING_UM:
                w = current_wear + wear_rate * step * (1.0 + 0.015 * step)
            else:
                w = current_wear + wear_rate * step
            proj_wear.append(round(float(w), 2))

        # 6. Explicit Mathematical Assumptions
        assumptions = [
            f"End-of-Life Flank Wear Criterion: VB_max = {self.wear_limit_um:.1f} µm (ISO 8688-2 / PHM standard).",
            f"Current Flank Wear: VB = {current_wear:.1f} µm (Remaining tolerance headroom: {wear_headroom:.1f} µm).",
            f"Estimated Wear Velocity: d(VB)/d(cut) = {wear_rate:.3f} µm/cut based on operational degradation phase.",
            f"Machining Pass Geometry: Single pass length = {cut_length_mm:.0f} mm at table feed rate = {feed_rate_mm_min:.0f} mm/min.",
            f"Cycle Time: {time_per_cut_min:.2f} min per milling pass including tool entry/retract.",
            "Confidence Margin: 82% lower bound (safety buffer for thermal spikes) to 118% upper bound."
        ]

        return {
            "rul_cuts": rul_cuts,
            "rul_cuts_lower": rul_lower,
            "rul_cuts_upper": rul_upper,
            "rul_minutes": rul_minutes,
            "rul_minutes_lower": rul_minutes_lower,
            "rul_minutes_upper": rul_minutes_upper,
            "wear_rate_um_per_cut": round(wear_rate, 3),
            "degradation_phase": phase,
            "phase_description": desc,
            "current_wear_um": round(current_wear, 2),
            "wear_headroom_um": round(wear_headroom, 2),
            "assumptions": assumptions,
            "trajectory_cuts": proj_cuts.tolist(),
            "trajectory_wear": proj_wear
        }
