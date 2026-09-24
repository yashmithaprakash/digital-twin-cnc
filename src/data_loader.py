"""
Data Loader and Storage module for CNC Digital Twin (PRJ_422).
Handles SQLite local database storage, automatic generation of realistic
synthetic CNC sensor data (PHM 2010 structure), and detection/parsing
of real uploaded CNC datasets.
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

from utils.config import (
    DB_PATH,
    SAMPLE_DATA_PATH,
    NOMINAL_SPINDLE_SPEED_RPM,
    NOMINAL_FEED_RATE_MM_MIN,
    NOMINAL_DEPTH_OF_CUT_MM,
    WEAR_MAX_THRESHOLD_UM,
    WEAR_INITIAL_UM,
    NUM_FLUTES
)


# ==============================================================================
# 1. SQLite Database Management
# ==============================================================================

def get_db_connection():
    """Create or connect to the local SQLite database safely."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables for telemetry, alerts, and maintenance logs safely."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Telemetry snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cut_id INTEGER,
                force_x REAL,
                force_y REAL,
                force_z REAL,
                force_resultant REAL,
                vibration_x REAL,
                vibration_y REAL,
                vibration_z REAL,
                vibration_rms REAL,
                ae_rms REAL,
                spindle_speed REAL,
                feed_rate REAL,
                depth_of_cut REAL,
                flank_wear_um REAL,
                tool_health_pct REAL,
                rul_cuts INTEGER,
                anomaly_status TEXT,
                data_source TEXT
            )
        """)

        # Maintenance events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                tool_id TEXT,
                flank_wear_at_service REAL,
                operating_cuts INTEGER,
                technician TEXT,
                action_taken TEXT,
                notes TEXT
            )
        """)

        # Anomaly alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cut_id INTEGER,
                severity TEXT,
                anomaly_score REAL,
                trigger_feature TEXT,
                description TEXT
            )
        """)

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Database initialization encountered an issue: {e}")


def log_telemetry(record: Dict[str, Any]):
    """Insert a single telemetry record into SQLite safely."""
    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO telemetry_logs (
                cut_id, force_x, force_y, force_z, force_resultant,
                vibration_x, vibration_y, vibration_z, vibration_rms, ae_rms,
                spindle_speed, feed_rate, depth_of_cut, flank_wear_um,
                tool_health_pct, rul_cuts, anomaly_status, data_source
            ) VALUES (
                :cut_id, :force_x, :force_y, :force_z, :force_resultant,
                :vibration_x, :vibration_y, :vibration_z, :vibration_rms, :ae_rms,
                :spindle_speed, :feed_rate, :depth_of_cut, :flank_wear_um,
                :tool_health_pct, :rul_cuts, :anomaly_status, :data_source
            )
        """, record)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Notice: Telemetry record logging skipped: {e}")


def log_maintenance_event(event_type: str, tool_id: str, flank_wear: float,
                          operating_cuts: int, technician: str, action: str, notes: str = ""):
    """Insert a maintenance event record safely."""
    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO maintenance_logs (
                event_type, tool_id, flank_wear_at_service, operating_cuts,
                technician, action_taken, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_type, tool_id, flank_wear, operating_cuts, technician, action, notes))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Notice: Maintenance event logging skipped: {e}")


def get_maintenance_history() -> pd.DataFrame:
    """Fetch all recorded maintenance events safely, returning empty DataFrame on any issue."""
    try:
        init_db()
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM maintenance_logs ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Notice: Could not query maintenance history ({e}), returning default empty frame.")
        return pd.DataFrame(columns=[
            "id", "timestamp", "event_type", "tool_id",
            "flank_wear_at_service", "operating_cuts", "technician", "action_taken", "notes"
        ])


def get_telemetry_history(limit: int = 200) -> pd.DataFrame:
    """Fetch recent telemetry logs safely."""
    try:
        init_db()
        conn = get_db_connection()
        df = pd.read_sql_query(f"SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Notice: Could not query telemetry history ({e}), returning default empty frame.")
        return pd.DataFrame(columns=[
            "id", "timestamp", "cut_id", "force_x", "force_y", "force_z", "force_resultant",
            "vibration_x", "vibration_y", "vibration_z", "vibration_rms", "ae_rms",
            "spindle_speed", "feed_rate", "depth_of_cut", "flank_wear_um",
            "tool_health_pct", "rul_cuts", "anomaly_status", "data_source"
        ])


# ==============================================================================
# 2. Synthetic CNC Data Generation (Realistic Physics-Grounded PHM 2010 Model)
# ==============================================================================

def generate_synthetic_cnc_dataset(num_cuts: int = 315, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic CNC milling sensor records matching the
    PHM Society 2010 CNC Milling Challenge specifications.
    
    Models:
    - 3-stage tool wear progression (Break-in -> Steady-state -> Accelerated tertiary wear)
    - Dynamic cutting forces (Fx, Fy, Fz) scaling with tool flank wear (VB)
    - Accelerometer vibration (Vx, Vy, Vz) scaling with chattering & edge dullness
    - Acoustic Emission (AE-RMS) sensitive to friction, micro-chipping, and rubbing
    - Realistic manufacturing anomalies (chatter, chip welding, micro-fracture)
    """
    np.random.seed(random_seed)
    cuts = np.arange(1, num_cuts + 1)
    t_norm = cuts / num_cuts  # Normalized progression [0, 1]

    # 1. Flank Wear Progression VB(t) (micrometers, µm)
    # Stage 1 (0-15%): Rapid initial break-in (20µm -> 45µm)
    # Stage 2 (15-80%): Linear steady-state wear (45µm -> 135µm)
    # Stage 3 (80-100%): Exponential accelerated tertiary breakdown (135µm -> 210µm)
    wear_curve = (
        WEAR_INITIAL_UM
        + 30.0 * (1 - np.exp(-t_norm / 0.08))  # Initial break-in exponential rise
        + 110.0 * t_norm                        # Steady-state linear slope
        + 55.0 * np.maximum(0, (t_norm - 0.78) / 0.22) ** 2.2 # Accelerated tertiary wear
    )
    # Add realistic measurement noise (microscope optical uncertainty ~ ±2.5 µm)
    wear_noise = np.random.normal(0, 2.2, size=num_cuts)
    flank_wear_um = np.maximum(15.0, wear_curve + wear_noise)
    # Enforce monotonic trend with small local noise
    flank_wear_um = np.maximum.accumulate(flank_wear_um - np.random.uniform(0, 1.2, size=num_cuts))

    # Tool Health Index (100% when brand new -> 0% when reaching WEAR_MAX_THRESHOLD_UM)
    tool_health_pct = np.clip(100.0 * (1.0 - (flank_wear_um / WEAR_MAX_THRESHOLD_UM)), 0.0, 100.0)

    # Remaining Useful Life (RUL) in cuts until wear reaches WEAR_MAX_THRESHOLD_UM
    wear_threshold = WEAR_MAX_THRESHOLD_UM
    rul_cuts = np.zeros(num_cuts, dtype=int)
    for i in range(num_cuts):
        # Remaining cuts index where wear crosses threshold
        future_breaches = np.where(flank_wear_um[i:] >= wear_threshold)[0]
        if len(future_breaches) > 0:
            rul_cuts[i] = int(future_breaches[0])
        else:
            # Estimate using local derivative
            slope = (flank_wear_um[i] - flank_wear_um[max(0, i - 10)]) / max(1, min(10, i))
            rul_cuts[i] = int(max(0, (wear_threshold - flank_wear_um[i]) / max(slope, 0.4)))

    # 2. Sensor Signatures (Physics-informed coupling with tool wear)
    # In milling, as flank wear VB grows:
    # - Cutting edge radius increases -> higher friction & plow forces (especially Fy & Fz)
    # - Dynamic chattering increases vibration energy in X and Y
    # - AE-RMS rises significantly due to sliding contact friction
    wear_ratio = flank_wear_um / 100.0  # normalized wear factor

    # Nominal baseline forces (Inconel 718 milling):
    # Fx (feed direction): ~110N baseline -> ~240N at failure
    # Fy (radial/normal direction): ~80N baseline -> ~260N at failure (heavily affected by flank rubbing)
    # Fz (axial direction): ~50N baseline -> ~140N at failure
    force_x = 110.0 + 75.0 * wear_ratio + np.random.normal(0, 8.5, size=num_cuts)
    force_y = 85.0 + 95.0 * (wear_ratio ** 1.15) + np.random.normal(0, 9.0, size=num_cuts)
    force_z = 52.0 + 48.0 * wear_ratio + np.random.normal(0, 5.5, size=num_cuts)
    force_resultant = np.sqrt(force_x**2 + force_y**2 + force_z**2)

    # Accelerometer Vibration (RMS in g):
    # Base vibration: ~0.8g -> ~3.6g at severe chattering / dull edge
    vib_x = 0.82 + 1.25 * (wear_ratio ** 1.3) + np.random.normal(0, 0.12, size=num_cuts)
    vib_y = 0.78 + 1.35 * (wear_ratio ** 1.35) + np.random.normal(0, 0.11, size=num_cuts)
    vib_z = 0.65 + 0.90 * (wear_ratio ** 1.2) + np.random.normal(0, 0.09, size=num_cuts)
    vibration_rms = np.sqrt((vib_x**2 + vib_y**2 + vib_z**2) / 3.0)

    # Acoustic Emission AE-RMS (Volts):
    # High frequency stress wave energy from micro-cracking and high-shear friction
    ae_rms = 0.18 + 0.32 * (wear_ratio ** 0.9) + 0.08 * (t_norm ** 2) + np.random.normal(0, 0.025, size=num_cuts)
    ae_rms = np.clip(ae_rms, 0.05, 1.2)

    # 3. Operating Process Parameters
    spindle_speed = np.full(num_cuts, NOMINAL_SPINDLE_SPEED_RPM) + np.random.normal(0, 15.0, size=num_cuts)
    feed_rate = np.full(num_cuts, NOMINAL_FEED_RATE_MM_MIN) + np.random.normal(0, 4.0, size=num_cuts)
    depth_of_cut = np.full(num_cuts, NOMINAL_DEPTH_OF_CUT_MM) + np.random.normal(0, 0.015, size=num_cuts)
    coolant = ["Flood"] * num_cuts

    # 4. Inject Realistic Anomalies (e.g., thermal spike at cut 120, chatter at 215, micro-chip at 270)
    is_anomaly = np.zeros(num_cuts, dtype=int)
    anomaly_types = ["NORMAL"] * num_cuts

    anomaly_indices = [85, 160, 245, 290]
    for idx in anomaly_indices:
        if idx < num_cuts:
            is_anomaly[idx] = 1
            if idx == 85:
                # Intermittent hard inclusion in workpiece -> force spike
                force_x[idx] += 85.0
                force_y[idx] += 90.0
                force_resultant[idx] = np.sqrt(force_x[idx]**2 + force_y[idx]**2 + force_z[idx]**2)
                anomaly_types[idx] = "Workpiece Hard Inclusion (Force Spike)"
            elif idx == 160:
                # Resonant Chattering event -> high vibration burst
                vib_x[idx] += 1.8
                vib_y[idx] += 2.1
                vibration_rms[idx] = np.sqrt((vib_x[idx]**2 + vib_y[idx]**2 + vib_z[idx]**2) / 3.0)
                anomaly_types[idx] = "Severe Chattering Resonance"
            elif idx == 245:
                # Tool Flute Micro-Chipping -> AE spike & vibration jump
                ae_rms[idx] += 0.38
                vib_y[idx] += 1.4
                vibration_rms[idx] = np.sqrt((vib_x[idx]**2 + vib_y[idx]**2 + vib_z[idx]**2) / 3.0)
                anomaly_types[idx] = "Cutting Flute Micro-Chipping"
            elif idx == 290:
                # Severe thermal breakdown & catastrophic rubbing
                force_y[idx] += 120.0
                ae_rms[idx] += 0.45
                force_resultant[idx] = np.sqrt(force_x[idx]**2 + force_y[idx]**2 + force_z[idx]**2)
                anomaly_types[idx] = "Thermal Breakdown & Edge Wear Collapse"

    # Assemble comprehensive DataFrame
    df = pd.DataFrame({
        "cut_id": cuts,
        "spindle_speed_rpm": np.round(spindle_speed, 1),
        "feed_rate_mm_min": np.round(feed_rate, 1),
        "depth_of_cut_mm": np.round(depth_of_cut, 3),
        "coolant": coolant,
        "force_x": np.round(force_x, 2),
        "force_y": np.round(force_y, 2),
        "force_z": np.round(force_z, 2),
        "force_resultant": np.round(force_resultant, 2),
        "vibration_x": np.round(vib_x, 3),
        "vibration_y": np.round(vib_y, 3),
        "vibration_z": np.round(vib_z, 3),
        "vibration_rms": np.round(vibration_rms, 3),
        "ae_rms": np.round(ae_rms, 4),
        "flank_wear_um": np.round(flank_wear_um, 2),
        "tool_health_pct": np.round(tool_health_pct, 1),
        "rul_cuts": rul_cuts,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_types
    })

    return df


def generate_cut_waveform(cut_id: int, flank_wear_um: float,
                          spindle_speed_rpm: float = NOMINAL_SPINDLE_SPEED_RPM,
                          feed_rate_mm_min: float = NOMINAL_FEED_RATE_MM_MIN,
                          num_points: int = 500) -> pd.DataFrame:
    """
    Generate high-resolution dynamic sensor waveforms for a single milling cut.
    Simulates the tooth-passing frequency excitation (4 flutes, ~693 Hz),
    random chip formation dynamics, and AE high-frequency bursts.
    """
    # Duration ~ 0.04 seconds (approx 7 full spindle rotations for detailed inspection)
    t = np.linspace(0, 0.04, num_points)
    tooth_passing_freq = (spindle_speed_rpm / 60.0) * NUM_FLUTES  # ~693.3 Hz
    spindle_freq = spindle_speed_rpm / 60.0                       # ~173.3 Hz

    wear_scale = 1.0 + (flank_wear_um / 150.0)

    # 1. Dynamic Force (Fx, Fy, Fz) with periodic tooth impact + cutter runout
    fx_mean = 110.0 * wear_scale
    fy_mean = 85.0 * (wear_scale ** 1.2)
    fz_mean = 52.0 * wear_scale

    fx_wave = (
        fx_mean
        + 35.0 * np.sin(2 * np.pi * tooth_passing_freq * t)
        + 18.0 * np.sin(4 * np.pi * tooth_passing_freq * t)
        + 12.0 * np.cos(2 * np.pi * spindle_freq * t)  # Runout modulation
        + np.random.normal(0, 6.0, num_points)
    )
    fy_wave = (
        fy_mean
        + 42.0 * np.cos(2 * np.pi * tooth_passing_freq * t)
        + 20.0 * np.sin(2 * np.pi * tooth_passing_freq * t + 0.8)
        + 15.0 * np.sin(2 * np.pi * spindle_freq * t)
        + np.random.normal(0, 7.0, num_points)
    )
    fz_wave = (
        fz_mean
        + 18.0 * np.sin(2 * np.pi * tooth_passing_freq * t - 0.5)
        + np.random.normal(0, 4.0, num_points)
    )
    f_res_wave = np.sqrt(fx_wave**2 + fy_wave**2 + fz_wave**2)

    # 2. Dynamic Accelerometer Signals (g)
    vib_amp = 0.8 * (wear_scale ** 1.35)
    vx_wave = (
        vib_amp * np.sin(2 * np.pi * tooth_passing_freq * t)
        + 0.6 * np.sin(2 * np.pi * 2150.0 * t)  # Structural resonance mode
        + np.random.normal(0, 0.25, num_points)
    )
    vy_wave = (
        vib_amp * 1.1 * np.cos(2 * np.pi * tooth_passing_freq * t)
        + 0.7 * np.sin(2 * np.pi * 2420.0 * t)
        + np.random.normal(0, 0.28, num_points)
    )
    vz_wave = (
        vib_amp * 0.7 * np.sin(2 * np.pi * tooth_passing_freq * t + 1.2)
        + np.random.normal(0, 0.20, num_points)
    )

    # 3. Dynamic Acoustic Emission (V)
    # High-frequency friction + intermittent bursts when chips shear off
    ae_baseline = 0.2 * wear_scale
    burst_mask = np.random.binomial(1, 0.08, num_points) * np.random.uniform(0.2, 0.5, num_points)
    ae_wave = ae_baseline + np.abs(np.random.normal(0, 0.08, num_points)) + burst_mask

    return pd.DataFrame({
        "time_ms": np.round(t * 1000, 3),
        "force_x": np.round(fx_wave, 2),
        "force_y": np.round(fy_wave, 2),
        "force_z": np.round(fz_wave, 2),
        "force_resultant": np.round(f_res_wave, 2),
        "vibration_x": np.round(vx_wave, 3),
        "vibration_y": np.round(vy_wave, 3),
        "vibration_z": np.round(vz_wave, 3),
        "ae_rms": np.round(ae_wave, 4)
    })


# ==============================================================================
# 3. Dataset Loading, Detection & Standardization
# ==============================================================================

def detect_and_standardize_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect column headers from real PHM Society 2010 CNC dataset or
    custom industrial telemetry files and normalize into standard schema.
    Accurately maps:
    - Tool wear: max_wear, flank_wear_um, flank_wear, tool_wear, vb, flute_1
    - Cutting forces: force_x_max, force_y_max, force_x_p2p, force_x, force_y, force_z, force
    - Vibrations: vib_x_std, vib_y_std, vib_x_max, vibration_x, vibration_y, vibration
    - Acoustic Emission: ae_rms, acoustic_emission
    - Process parameters: spindle_speed, feed_rate, depth_of_cut, coolant
    - Cut index & Cutter ID: cut, cut_number, cut_id, cutter, cutter_id
    """
    df = raw_df.copy()
    col_map = {}

    lower_cols = {c.lower().strip(): c for c in df.columns}

    # Map Cut ID
    for candidate in ["cut_id", "cut", "cut_number", "cut_no", "cut_idx", "cut_index"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "cut_id"
            break

    # Map Cutter ID (c1, c4, c6, etc.)
    for candidate in ["cutter_id", "cutter", "tool_id", "cutter_name"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "cutter_id"
            break

    # Map Operating Conditions
    for candidate in ["spindle_speed_rpm", "spindle_speed", "speed", "rpm"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "spindle_speed_rpm"
            break
    for candidate in ["feed_rate_mm_min", "feed_rate", "feed", "f_mm_min"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "feed_rate_mm_min"
            break
    for candidate in ["depth_of_cut_mm", "depth_of_cut", "doc", "ap"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "depth_of_cut_mm"
            break
    for candidate in ["coolant", "coolant_status"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "coolant"
            break

    # Map Tool Wear
    for candidate in ["flank_wear_um", "max_wear", "tool_wear", "flank_wear", "vb", "wear", "flank wear", "flank_wear_mm", "flute_1"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "flank_wear_um"
            break

    # Map Tool Health & RUL if present
    for candidate in ["tool_health_pct", "tool_health", "health_pct", "health"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "tool_health_pct"
            break
    for candidate in ["rul_cuts", "rul", "remaining_useful_life"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "rul_cuts"
            break

    # Map Force
    for candidate in ["force_x", "fx", "force_x_max", "force_x_p2p", "force_x_n", "smcac", "force x"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "force_x"
            break
    for candidate in ["force_y", "fy", "force_y_max", "force_y_mean", "force_y_n", "smcdc", "force y"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "force_y"
            break
    for candidate in ["force_z", "fz", "force_z_n", "force z"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "force_z"
            break
    for candidate in ["force_resultant", "force", "f_res", "fres", "resultant_force"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "force_resultant"
            break

    # Map Vibration
    for candidate in ["vibration_x", "vx", "vib_x_std", "vib_x", "vib_x_max", "vib_table", "vibration x"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "vibration_x"
            break
    for candidate in ["vibration_y", "vy", "vib_y_std", "vib_y", "vib_spindle", "vibration y"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "vibration_y"
            break
    for candidate in ["vibration_z", "vz", "vib_z", "vibration z"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "vibration_z"
            break
    for candidate in ["vibration_rms", "vibration", "vib_rms", "rms_vibration"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "vibration_rms"
            break

    # Map Acoustic Emission
    for candidate in ["ae_rms", "ae", "acoustic_emission", "ae_spindle", "ae_table"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "ae_rms"
            break

    df = df.rename(columns=col_map)

    # If flank wear was in mm, convert to micrometers (if max < 2.0)
    if "flank_wear_um" in df.columns:
        if df["flank_wear_um"].max() < 2.0:
            df["flank_wear_um"] = df["flank_wear_um"] * 1000.0

    # Ensure essential columns exist, fill sensible defaults if absent
    if "cut_id" not in df.columns:
        df["cut_id"] = np.arange(1, len(df) + 1)
    if "spindle_speed_rpm" not in df.columns:
        df["spindle_speed_rpm"] = NOMINAL_SPINDLE_SPEED_RPM
    if "feed_rate_mm_min" not in df.columns:
        df["feed_rate_mm_min"] = NOMINAL_FEED_RATE_MM_MIN
    if "depth_of_cut_mm" not in df.columns:
        df["depth_of_cut_mm"] = NOMINAL_DEPTH_OF_CUT_MM
    if "coolant" not in df.columns:
        df["coolant"] = "Off"

    # Compute resultant force if missing
    if "force_resultant" not in df.columns:
        if "force_x" in df.columns and "force_y" in df.columns:
            fz = df["force_z"] if "force_z" in df.columns else 0.0
            df["force_resultant"] = np.sqrt(df["force_x"]**2 + df["force_y"]**2 + fz**2)
        else:
            df["force_resultant"] = 150.0

    # Ensure force_x and force_y exist
    if "force_x" not in df.columns:
        df["force_x"] = df["force_resultant"] * 0.75
    if "force_y" not in df.columns:
        df["force_y"] = df["force_resultant"] * 0.65
    if "force_z" not in df.columns:
        df["force_z"] = df["force_resultant"] * 0.35

    # Compute vibration RMS if missing
    if "vibration_rms" not in df.columns:
        if "vibration_x" in df.columns and "vibration_y" in df.columns:
            vz = df["vibration_z"] if "vibration_z" in df.columns else 0.0
            df["vibration_rms"] = np.sqrt((df["vibration_x"]**2 + df["vibration_y"]**2 + vz**2) / 3.0)
        else:
            df["vibration_rms"] = 1.0

    if "vibration_x" not in df.columns:
        df["vibration_x"] = df["vibration_rms"] * 0.9
    if "vibration_y" not in df.columns:
        df["vibration_y"] = df["vibration_rms"] * 0.95
    if "vibration_z" not in df.columns:
        df["vibration_z"] = df["vibration_rms"] * 0.65

    if "ae_rms" not in df.columns:
        # Realistic proxy based on wear and vibration
        if "flank_wear_um" in df.columns:
            df["ae_rms"] = np.clip(0.35 + 0.002 * df["flank_wear_um"] + 0.1 * df["vibration_rms"], 0.1, 1.2)
        else:
            df["ae_rms"] = 0.35

    # Compute tool health and RUL if flank_wear_um exists
    if "flank_wear_um" in df.columns:
        if "tool_health_pct" not in df.columns:
            df["tool_health_pct"] = np.clip(100.0 * (1.0 - (df["flank_wear_um"] / WEAR_MAX_THRESHOLD_UM)), 0.0, 100.0)
        if "rul_cuts" not in df.columns:
            rul_list = []
            for i in range(len(df)):
                future = np.where(df["flank_wear_um"].iloc[i:] >= WEAR_MAX_THRESHOLD_UM)[0]
                rul_list.append(int(future[0]) if len(future) > 0 else max(0, len(df) - i))
            df["rul_cuts"] = rul_list
    else:
        df["flank_wear_um"] = 50.0
        df["tool_health_pct"] = 75.0
        df["rul_cuts"] = 100

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = np.where(df["flank_wear_um"] >= WEAR_MAX_THRESHOLD_UM, 1, 0)
    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = np.where(df["flank_wear_um"] >= WEAR_MAX_THRESHOLD_UM, "Critical Flank Wear Exceeded", "NORMAL")

    return df


def get_available_cutters(df: pd.DataFrame) -> List[str]:
    """Return list of distinct cutter IDs in dataset (e.g. ['c1', 'c4', 'c6'])."""
    if "cutter_id" in df.columns:
        return [str(c) for c in df["cutter_id"].dropna().unique().tolist()]
    return []


def load_dataset(custom_file=None,
                 force_synthetic: bool = False,
                 selected_cutter: Optional[str] = None) -> Tuple[pd.DataFrame, bool, str]:
    """
    Load CNC dataset.
    Returns:
        (df, is_real_data, data_source_label)

    Priority:
    1. Custom uploaded file (if provided)
    2. Fallback to synthetic if force_synthetic is True
    3. Real PHM Society 2010 CNC Milling Dataset (from data/phm2010_milling_full.csv or features)
    4. Fallback to synthetic sample CNC data
    """
    from utils.config import (
        PHM_FULL_DATA_PATH,
        PHM_FEATURES_DATA_PATH,
        DATA_SOURCE_REAL_PHM,
        DATA_SOURCE_SYNTHETIC
    )

    # 1. Custom uploaded file
    if custom_file is not None:
        try:
            raw = pd.read_csv(custom_file)
            standardized = detect_and_standardize_data(raw)
            return standardized, True, f"REAL DATA (Uploaded: {getattr(custom_file, 'name', 'custom.csv')})"
        except Exception as e:
            print(f"Error parsing custom uploaded file: {e}")

    # 2. Forced synthetic mode
    if force_synthetic:
        if SAMPLE_DATA_PATH.exists():
            df = pd.read_csv(SAMPLE_DATA_PATH)
            return df, False, DATA_SOURCE_SYNTHETIC
        df = generate_synthetic_cnc_dataset(num_cuts=315)
        df.to_csv(SAMPLE_DATA_PATH, index=False)
        return df, False, DATA_SOURCE_SYNTHETIC

    # 3. Check for Real PHM 2010 Full Dataset
    if PHM_FULL_DATA_PATH.exists():
        try:
            raw = pd.read_csv(PHM_FULL_DATA_PATH)
            standardized = detect_and_standardize_data(raw)
            if selected_cutter and selected_cutter != "All" and "cutter_id" in standardized.columns:
                standardized = standardized[standardized["cutter_id"] == selected_cutter].copy()
                label = f"{DATA_SOURCE_REAL_PHM} (Cutter {selected_cutter.upper()})"
            else:
                label = DATA_SOURCE_REAL_PHM
            return standardized, True, label
        except Exception as e:
            print(f"Error loading PHM full dataset: {e}")

    # 4. Check for Real PHM 2010 Features Dataset
    if PHM_FEATURES_DATA_PATH.exists():
        try:
            raw = pd.read_csv(PHM_FEATURES_DATA_PATH)
            standardized = detect_and_standardize_data(raw)
            if selected_cutter and selected_cutter != "All" and "cutter_id" in standardized.columns:
                standardized = standardized[standardized["cutter_id"] == selected_cutter].copy()
                label = f"{DATA_SOURCE_REAL_PHM} (Cutter {selected_cutter.upper()})"
            else:
                label = DATA_SOURCE_REAL_PHM
            return standardized, True, label
        except Exception as e:
            print(f"Error loading PHM features dataset: {e}")

    # 5. Check any *phm*.csv in data/
    phm_candidates = list(Path(SAMPLE_DATA_PATH.parent).glob("*phm*.csv"))
    if phm_candidates:
        try:
            raw = pd.read_csv(phm_candidates[0])
            standardized = detect_and_standardize_data(raw)
            return standardized, True, DATA_SOURCE_REAL_PHM
        except Exception as e:
            print(f"Error loading PHM candidate: {e}")

    # 6. Fallback: Synthetic sample CNC data
    if SAMPLE_DATA_PATH.exists():
        try:
            df = pd.read_csv(SAMPLE_DATA_PATH)
            return df, False, DATA_SOURCE_SYNTHETIC
        except Exception as e:
            print(f"Error reading existing sample data: {e}")

    # Generate fresh synthetic dataset as last resort fallback
    df = generate_synthetic_cnc_dataset(num_cuts=315)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return df, False, DATA_SOURCE_SYNTHETIC
