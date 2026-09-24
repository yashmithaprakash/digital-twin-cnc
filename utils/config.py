"""
Configuration module for the Digital Twin CNC Manufacturing System (PRJ_422).
Defines machine physical limits, tool wear thresholds, model parameters,
database paths, and UI color palettes.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data" / "digital_twin_cnc.db"
SAMPLE_DATA_PATH = DATA_DIR / "sample_cnc_data.csv"
PHM_FULL_DATA_PATH = DATA_DIR / "phm2010_milling_full.csv"
PHM_FEATURES_DATA_PATH = DATA_DIR / "phm2010_milling_features.csv"
WEAR_MODEL_PATH = MODELS_DIR / "tool_wear_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"

# Real dataset identifier constant
DATA_SOURCE_REAL_PHM = "PHM Society 2010 CNC Milling Dataset"
DATA_SOURCE_SYNTHETIC = "High-Fidelity Synthetic CNC Twin (Fallback)"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# CNC Milling Physical & Nominal Parameters
NOMINAL_SPINDLE_SPEED_RPM = 10400.0  # RPM (PHM 2010 nominal speed)
NOMINAL_FEED_RATE_MM_MIN = 1555.0    # mm/min
NOMINAL_DEPTH_OF_CUT_MM = 1.5        # mm
NOMINAL_TOOL_DIAMETER_MM = 6.0       # mm (4-flute tungsten carbide cutter)
NUM_FLUTES = 4

# Tool Wear Limits (Flank Wear VB in micrometers, µm)
# According to ISO 8688 and PHM 2010 challenge criteria
WEAR_INITIAL_UM = 20.0        # Initial break-in wear
WEAR_WARNING_UM = 120.0       # Yellow alert: tool dressing/inspection recommended
WEAR_CRITICAL_UM = 165.0      # Orange alert: maintenance scheduling
WEAR_MAX_THRESHOLD_UM = 195.0 # Red alert: immediate tool replacement limit

# Health status labels and threshold mapping
STATUS_NORMAL = "NORMAL"
STATUS_WARNING = "WARNING"
STATUS_ANOMALY = "ANOMALY"
STATUS_CRITICAL = "CRITICAL"

# Thresholds for Sensor Readings (Engineering Units)
LIMITS = {
    "force_resultant_warning": 280.0,    # Newtons (N)
    "force_resultant_critical": 380.0,   # Newtons (N)
    "vibration_rms_warning": 2.2,        # g (acceleration)
    "vibration_rms_critical": 3.8,       # g
    "ae_rms_warning": 0.45,              # Volts (AE RMS)
    "ae_rms_critical": 0.75,             # Volts
}

# UI Theme Color Palettes
COLORS = {
    "normal": "#10B981",     # Emerald green
    "warning": "#F59E0B",    # Amber/Yellow
    "critical": "#EF4444",   # Crimson Red
    "primary": "#3B82F6",    # Industrial Blue
    "secondary": "#6366F1",  # Indigo
    "dark_bg": "#0F172A",    # Dark Slate
    "card_bg": "#1E293B",    # Slate 800
    "accent": "#06B6D4",     # Cyan
    "text_muted": "#94A3B8", # Slate 400
}

# Supported Workpiece Materials for Simulation
MATERIALS = {
    "Inconel 718": {
        "machinability_index": 0.35,
        "specific_cutting_force_kc1": 2850,  # N/mm²
        "taylor_c": 65.0,
        "taylor_n": 0.22,
        "feed_exponent_a": 0.45,
        "doc_exponent_b": 0.30
    },
    "Titanium Ti-6Al-4V": {
        "machinability_index": 0.40,
        "specific_cutting_force_kc1": 2100,
        "taylor_c": 80.0,
        "taylor_n": 0.25,
        "feed_exponent_a": 0.40,
        "doc_exponent_b": 0.28
    },
    "Stainless Steel 316L": {
        "machinability_index": 0.55,
        "specific_cutting_force_kc1": 1950,
        "taylor_c": 140.0,
        "taylor_n": 0.30,
        "feed_exponent_a": 0.38,
        "doc_exponent_b": 0.25
    },
    "Aluminum 6061-T6": {
        "machinability_index": 1.0,
        "specific_cutting_force_kc1": 750,
        "taylor_c": 450.0,
        "taylor_n": 0.45,
        "feed_exponent_a": 0.30,
        "doc_exponent_b": 0.20
    }
}
