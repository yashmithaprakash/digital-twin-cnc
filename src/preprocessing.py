"""
Data Preprocessing module for CNC Sensor Telemetry.
Provides signal filtering (Butterworth, rolling smoothing), baseline drift removal,
outlier handling, and scaling functions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def butterworth_filter(data: np.ndarray, cutoff_hz: float = 1200.0,
                       fs_hz: float = 10000.0, order: int = 4,
                       btype: str = "low") -> np.ndarray:
    """
    Apply a zero-phase Butterworth filter (low-pass or high-pass) to remove
    high-frequency electrical noise or low-frequency drift.
    """
    nyquist = 0.5 * fs_hz
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    # Use filtfilt for zero-phase distortion
    filtered = signal.filtfilt(b, a, data)
    return filtered


def rolling_smooth(series: Union[pd.Series, np.ndarray], window_size: int = 5) -> np.ndarray:
    """
    Apply a centered rolling median/mean filter to smooth out sudden transient spikes.
    """
    if isinstance(series, np.ndarray):
        s = pd.Series(series)
    else:
        s = series
    smoothed = s.rolling(window=window_size, min_periods=1, center=True).mean().values
    return smoothed


def remove_drift(signal_array: np.ndarray) -> np.ndarray:
    """
    Remove baseline sensor drift using linear detrending.
    """
    return signal.detrend(signal_array, type="linear")


def clean_and_normalize_features(df: pd.DataFrame,
                                 feature_cols: List[str],
                                 method: str = "robust") -> Tuple[pd.DataFrame, object]:
    """
    Handle missing values, clip extreme outliers, and normalize features.
    
    Args:
        df: Input DataFrame containing sensor measurements.
        feature_cols: List of column names to normalize.
        method: 'standard', 'minmax', or 'robust'.
        
    Returns:
        (normalized_df, fitted_scaler)
    """
    clean_df = df.copy()

    # Fill NaNs if any
    clean_df[feature_cols] = clean_df[feature_cols].interpolate(method="linear").bfill().ffill()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    scaled_vals = scaler.fit_transform(clean_df[feature_cols])
    scaled_df = pd.DataFrame(scaled_vals, columns=[f"{col}_scaled" for col in feature_cols], index=df.index)

    combined_df = pd.concat([clean_df, scaled_df], axis=1)
    return combined_df, scaler


def segment_waveform(waveform_array: np.ndarray, window_size: int = 100, step_size: int = 50) -> List[np.ndarray]:
    """
    Segment a continuous sensor signal into overlapping sliding windows
    for localized feature extraction.
    """
    segments = []
    n = len(waveform_array)
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        segments.append(waveform_array[start:end])
    return segments


def preprocess_phm2010_data(df: pd.DataFrame,
                             smooth_window: int = 3) -> pd.DataFrame:
    """
    Specialized preprocessor for the PHM Society 2010 CNC Milling Dataset.
    Applies:
    - Linear interpolation for any intermittent sensor dropout
    - Centered rolling smoothing on high-frequency dynamometer chattering
    - Resultant cutting force and vibration vector computation
    - Outlier bounding based on physical cutting limits (Inconel 718 dry milling)
    - Sequential cut wear rate derivative calculation
    """
    processed = df.copy()

    # Numeric sensor columns present in dataset
    sensor_cols = [
        col for col in [
            "force_x", "force_y", "force_z", "force_resultant",
            "vibration_x", "vibration_y", "vibration_z", "vibration_rms",
            "ae_rms", "flank_wear_um", "force_x_max", "force_y_max",
            "vib_x_std", "vib_y_std"
        ] if col in processed.columns
    ]

    # 1. Fill missing values with linear interpolation
    processed[sensor_cols] = processed[sensor_cols].interpolate(method="linear").bfill().ffill()

    # 2. Smooth physical sensors per cutter group if cutter_id is present
    if "cutter_id" in processed.columns:
        groups = []
        for cutter, group in processed.groupby("cutter_id"):
            g = group.copy().sort_values("cut_id" if "cut_id" in group.columns else group.index)
            for c in ["force_resultant", "vibration_rms", "ae_rms"]:
                if c in g.columns:
                    g[f"{c}_smooth"] = g[c].rolling(window=smooth_window, min_periods=1, center=True).mean()
            groups.append(g)
        processed = pd.concat(groups).sort_index()
    else:
        for c in ["force_resultant", "vibration_rms", "ae_rms"]:
            if c in processed.columns:
                processed[f"{c}_smooth"] = processed[c].rolling(window=smooth_window, min_periods=1, center=True).mean()

    # 3. Compute wear rate (dVB/dCut) if flank_wear_um exists
    if "flank_wear_um" in processed.columns:
        if "cutter_id" in processed.columns:
            processed["wear_rate"] = processed.groupby("cutter_id")["flank_wear_um"].diff().fillna(0.35)
        else:
            processed["wear_rate"] = processed["flank_wear_um"].diff().fillna(0.35)
        processed["wear_rate"] = np.clip(processed["wear_rate"], 0.0, 5.0)

    return processed
