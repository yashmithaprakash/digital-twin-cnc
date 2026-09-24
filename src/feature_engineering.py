"""
Feature Engineering module for CNC Milling Digital Twin.
Extracts comprehensive time-domain and frequency-domain features from raw Force,
Vibration, and Acoustic Emission signals, as well as physical machining interaction indices.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any


def extract_time_domain_features(signal_array: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Extract statistical time-domain features from a 1D sensor signal array.
    Features:
    - Mean
    - Standard Deviation (Std)
    - Variance
    - Root Mean Square (RMS)
    - Peak (Max absolute amplitude)
    - Peak-to-Peak (Range)
    - Crest Factor (Peak / RMS)
    - Kurtosis (Peak sharpness, impact sensitivity)
    - Skewness (Signal asymmetry)
    - Shape Factor (RMS / Abs Mean)
    - Impulse Factor (Peak / Abs Mean)
    """
    arr = np.asarray(signal_array, dtype=float)
    if len(arr) == 0:
        return {}

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    var_val = float(np.var(arr))
    rms_val = float(np.sqrt(np.mean(arr ** 2)))
    peak_val = float(np.max(np.abs(arr)))
    p2p_val = float(np.ptp(arr))
    abs_mean = float(np.mean(np.abs(arr))) or 1e-6

    crest_factor = float(peak_val / (rms_val + 1e-6))
    kurtosis_val = float(stats.kurtosis(arr))
    skewness_val = float(stats.skew(arr))
    shape_factor = float(rms_val / abs_mean)
    impulse_factor = float(peak_val / abs_mean)

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mean": mean_val,
        f"{p}std": std_val,
        f"{p}var": var_val,
        f"{p}rms": rms_val,
        f"{p}peak": peak_val,
        f"{p}p2p": p2p_val,
        f"{p}crest_factor": crest_factor,
        f"{p}kurtosis": kurtosis_val,
        f"{p}skewness": skewness_val,
        f"{p}shape_factor": shape_factor,
        f"{p}impulse_factor": impulse_factor
    }


def extract_frequency_domain_features(signal_array: np.ndarray, fs_hz: float = 10000.0,
                                      prefix: str = "") -> Dict[str, float]:
    """
    Extract spectral features via Fast Fourier Transform (FFT).
    Features:
    - Dominant Peak Frequency (Hz)
    - Total Spectral Energy
    - Spectral Centroid (Center of gravity of spectrum)
    - Spectral Spread (Standard deviation of spectral distribution)
    """
    arr = np.asarray(signal_array, dtype=float)
    n = len(arr)
    if n < 8:
        return {}

    # Detrend before FFT to suppress DC leak
    arr_detrended = arr - np.mean(arr)
    fft_vals = np.abs(np.fft.rfft(arr_detrended))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    spectral_energy = float(np.sum(fft_vals ** 2) / n)
    dominant_freq = float(freqs[np.argmax(fft_vals)])

    sum_fft = np.sum(fft_vals) or 1e-6
    spectral_centroid = float(np.sum(freqs * fft_vals) / sum_fft)
    spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_vals) / sum_fft))

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}spectral_energy": spectral_energy,
        f"{p}dominant_freq_hz": dominant_freq,
        f"{p}spectral_centroid_hz": spectral_centroid,
        f"{p}spectral_spread_hz": spectral_spread
    }


def extract_features_from_cut_waveform(waveform_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract complete multi-sensor feature dictionary from a cut's raw waveform.
    Analyzes Force X/Y/Z, Vibration X/Y/Z, and Acoustic Emission.
    """
    features = {}

    # Force channels
    for col, prefix in [("force_x", "fx"), ("force_y", "fy"), ("force_z", "fz"), ("force_resultant", "f_res")]:
        if col in waveform_df.columns:
            arr = waveform_df[col].values
            features.update(extract_time_domain_features(arr, prefix=prefix))
            features.update(extract_frequency_domain_features(arr, prefix=prefix))

    # Vibration channels
    for col, prefix in [("vibration_x", "vx"), ("vibration_y", "vy"), ("vibration_z", "vz")]:
        if col in waveform_df.columns:
            arr = waveform_df[col].values
            features.update(extract_time_domain_features(arr, prefix=prefix))
            features.update(extract_frequency_domain_features(arr, prefix=prefix))

    # Acoustic emission
    if "ae_rms" in waveform_df.columns:
        arr = waveform_df["ae_rms"].values
        features.update(extract_time_domain_features(arr, prefix="ae"))

    # Machining physical ratios:
    # 1. Radial-to-feed force ratio (Fy / Fx): increases as flank wear VB grows and relief rubs
    fx_mean = features.get("fx_mean", 100.0)
    fy_mean = features.get("fy_mean", 80.0)
    features["force_ratio_fy_fx"] = float(fy_mean / (abs(fx_mean) + 1e-5))

    # 2. Total vibration power
    vx_rms = features.get("vx_rms", 1.0)
    vy_rms = features.get("vy_rms", 1.0)
    vz_rms = features.get("vz_rms", 0.8)
    features["vib_total_power"] = float(np.sqrt(vx_rms**2 + vy_rms**2 + vz_rms**2))

    return features


def extract_cut_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a dataset of cuts (or time-indexed records), compute engineered features
    ready for Scikit-learn model training and real-time inference.
    Supports both real PHM Society 2010 Milling Dataset and synthetic twin data.
    """
    feat_df = df.copy()

    # 1. Resultant cutting force
    if "force_resultant" not in feat_df.columns:
        fz = feat_df["force_z"] if "force_z" in feat_df.columns else 0.0
        feat_df["force_resultant"] = np.sqrt(feat_df["force_x"]**2 + feat_df["force_y"]**2 + fz**2)

    # 2. Force ratio Fy / Fx (sensitive to flank wear land rubbing and relief angle reduction)
    if "force_x" in feat_df.columns and "force_y" in feat_df.columns:
        feat_df["force_ratio_fy_fx"] = np.where(
            feat_df["force_x"].abs() > 1e-4,
            feat_df["force_y"] / feat_df["force_x"],
            1.0
        )
    else:
        feat_df["force_ratio_fy_fx"] = 0.85

    # 3. Vibration RMS magnitude
    if "vibration_rms" not in feat_df.columns:
        vz = feat_df["vibration_z"] if "vibration_z" in feat_df.columns else 0.0
        feat_df["vibration_rms"] = np.sqrt(
            (feat_df["vibration_x"]**2 + feat_df["vibration_y"]**2 + vz**2) / 3.0
        )

    # 4. Energy interaction terms (Force * Vibration * AE proxy)
    ae_col = feat_df["ae_rms"] if "ae_rms" in feat_df.columns else 0.3
    feat_df["energy_index"] = (
        (feat_df["force_resultant"] / 100.0) *
        feat_df["vibration_rms"] *
        (1.0 + ae_col * 2.0)
    )

    # 5. Cutting Power Proxy (Watts): P = (F_res * pi * D * N) / (60 * 1000)
    spindle = feat_df["spindle_speed_rpm"] if "spindle_speed_rpm" in feat_df.columns else 10400.0
    vc_m_min = (np.pi * 6.0 * spindle) / 1000.0  # cutting speed m/min for 6mm cutter
    feat_df["cutting_power_w"] = (feat_df["force_resultant"] * (vc_m_min / 60.0))

    # 6. Flank Rubbing Friction Index: Fy * feed_rate
    feed = feat_df["feed_rate_mm_min"] if "feed_rate_mm_min" in feat_df.columns else 1555.0
    fy_val = feat_df["force_y"] if "force_y" in feat_df.columns else (feat_df["force_resultant"] * 0.65)
    feat_df["friction_load_index"] = (fy_val * (feed / 1000.0))

    # 7. Rolling statistical trend features (grouped by cutter_id if available)
    if "cutter_id" in feat_df.columns:
        feat_df["force_res_rolling_mean_5"] = (
            feat_df.groupby("cutter_id")["force_resultant"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
        )
        feat_df["vib_rms_rolling_mean_5"] = (
            feat_df.groupby("cutter_id")["vibration_rms"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
        )
        if "ae_rms" in feat_df.columns:
            feat_df["ae_rolling_mean_5"] = (
                feat_df.groupby("cutter_id")["ae_rms"]
                .transform(lambda s: s.rolling(5, min_periods=1).mean())
            )
        else:
            feat_df["ae_rolling_mean_5"] = 0.35
        feat_df["force_diff"] = feat_df.groupby("cutter_id")["force_resultant"].diff().fillna(0)
        feat_df["vib_diff"] = feat_df.groupby("cutter_id")["vibration_rms"].diff().fillna(0)
    else:
        feat_df["force_res_rolling_mean_5"] = feat_df["force_resultant"].rolling(5, min_periods=1).mean()
        feat_df["vib_rms_rolling_mean_5"] = feat_df["vibration_rms"].rolling(5, min_periods=1).mean()
        if "ae_rms" in feat_df.columns:
            feat_df["ae_rolling_mean_5"] = feat_df["ae_rms"].rolling(5, min_periods=1).mean()
        else:
            feat_df["ae_rolling_mean_5"] = 0.35
        feat_df["force_diff"] = feat_df["force_resultant"].diff().fillna(0)
        feat_df["vib_diff"] = feat_df["vibration_rms"].diff().fillna(0)

    return feat_df


# The core primary features used by the Tool Wear & Anomaly Models
CORE_MODEL_FEATURES: List[str] = [
    "force_x",
    "force_y",
    "force_z",
    "force_resultant",
    "force_ratio_fy_fx",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "vibration_rms",
    "ae_rms",
    "energy_index",
    "cutting_power_w",
    "friction_load_index",
    "force_res_rolling_mean_5",
    "vib_rms_rolling_mean_5",
    "ae_rolling_mean_5",
    "spindle_speed_rpm",
    "feed_rate_mm_min",
    "depth_of_cut_mm"
]
