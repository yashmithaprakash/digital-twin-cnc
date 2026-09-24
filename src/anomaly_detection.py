"""
Machine Learning Anomaly Detection for CNC Milling Process.
Implements Scikit-learn IsolationForest combined with multi-sensor process
monitoring (Force, Vibration, Acoustic Emission, Spindle Speed, Feed Rate, Depth of Cut).
Classifies machine condition strictly into:
  - NORMAL
  - WARNING
  - ANOMALY
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils.config import (
    ANOMALY_MODEL_PATH,
    LIMITS,
    STATUS_NORMAL,
    STATUS_WARNING,
    STATUS_ANOMALY,
    STATUS_CRITICAL,
    NOMINAL_SPINDLE_SPEED_RPM,
    NOMINAL_FEED_RATE_MM_MIN,
    NOMINAL_DEPTH_OF_CUT_MM
)
from src.feature_engineering import extract_cut_level_features


# Input features spanning all 6 sensor & process categories:
# 1. Force (Resultant, Fx, Fy, Fz)
# 2. Vibration (RMS, Vx, Vy, Vz)
# 3. Acoustic Emission (AE-RMS)
# 4. Spindle Speed (RPM)
# 5. Feed Rate (mm/min)
# 6. Depth of Cut (mm)
ANOMALY_FEATURE_COLS = [
    "force_resultant",
    "force_x",
    "force_y",
    "force_z",
    "vibration_rms",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "ae_rms",
    "spindle_speed_rpm",
    "feed_rate_mm_min",
    "depth_of_cut_mm"
]

SENSOR_METADATA = {
    "force_resultant": {"label": "Resultant Force (F_res)", "unit": "N", "nominal": 155.0, "warn_thresh": LIMITS["force_resultant_warning"], "crit_thresh": LIMITS["force_resultant_critical"]},
    "force_x": {"label": "Cutting Force (Fx)", "unit": "N", "nominal": 125.0, "warn_thresh": 240.0, "crit_thresh": 310.0},
    "force_y": {"label": "Feed Force (Fy)", "unit": "N", "nominal": 105.0, "warn_thresh": 220.0, "crit_thresh": 290.0},
    "force_z": {"label": "Axial Force (Fz)", "unit": "N", "nominal": 65.0, "warn_thresh": 120.0, "crit_thresh": 160.0},
    "vibration_rms": {"label": "Vibration RMS", "unit": "g", "nominal": 1.15, "warn_thresh": LIMITS["vibration_rms_warning"], "crit_thresh": LIMITS["vibration_rms_critical"]},
    "vibration_x": {"label": "Vibration Vx", "unit": "g", "nominal": 1.10, "warn_thresh": 2.2, "crit_thresh": 3.6},
    "vibration_y": {"label": "Vibration Vy", "unit": "g", "nominal": 1.05, "warn_thresh": 2.3, "crit_thresh": 3.7},
    "vibration_z": {"label": "Vibration Vz", "unit": "g", "nominal": 0.85, "warn_thresh": 1.8, "crit_thresh": 2.8},
    "ae_rms": {"label": "Acoustic Emission (AE-RMS)", "unit": "V", "nominal": 0.28, "warn_thresh": LIMITS["ae_rms_warning"], "crit_thresh": LIMITS["ae_rms_critical"]},
    "spindle_speed_rpm": {"label": "Spindle Speed", "unit": "RPM", "nominal": NOMINAL_SPINDLE_SPEED_RPM, "warn_thresh": NOMINAL_SPINDLE_SPEED_RPM * 1.06, "crit_thresh": NOMINAL_SPINDLE_SPEED_RPM * 1.12},
    "feed_rate_mm_min": {"label": "Feed Rate", "unit": "mm/min", "nominal": NOMINAL_FEED_RATE_MM_MIN, "warn_thresh": NOMINAL_FEED_RATE_MM_MIN * 1.08, "crit_thresh": NOMINAL_FEED_RATE_MM_MIN * 1.15},
    "depth_of_cut_mm": {"label": "Depth of Cut", "unit": "mm", "nominal": NOMINAL_DEPTH_OF_CUT_MM, "warn_thresh": NOMINAL_DEPTH_OF_CUT_MM * 1.10, "crit_thresh": NOMINAL_DEPTH_OF_CUT_MM * 1.25}
}


class CNCAnomalyDetector:
    """
    Scikit-learn IsolationForest Anomaly Detector with CNC Process Explainability:
    Combines multivariate IsolationForest unsupervised learning with
    deterministic CNC physical process threshold monitoring.
    Classifies observations strictly into NORMAL, WARNING, and ANOMALY.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = ANOMALY_FEATURE_COLS
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.is_trained: bool = False

    def train(self, df: pd.DataFrame):
        """
        Fit Isolation Forest on nominal and representative sensor operating data.
        Computes baseline distribution metrics for explainability.
        """
        feat_df = extract_cut_level_features(df)
        available = [c for c in self.feature_names if c in feat_df.columns]
        if len(available) < 4:
            raise ValueError("Insufficient feature columns to train anomaly detector.")
        self.feature_names = available

        # Compute baseline statistics for explainable attribution
        self.baseline_stats = {}
        for col in self.feature_names:
            series = feat_df[col].dropna()
            self.baseline_stats[col] = {
                "mean": float(series.mean()),
                "std": float(max(1e-4, series.std())),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "min": float(series.min()),
                "max": float(series.max())
            }

        X = feat_df[self.feature_names].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=120,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        self.is_trained = True

    def score_samples(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for a batch of records.
        Returns:
            (anomaly_labels, normalized_scores)
            anomaly_labels: 1 (normal/inlier), -1 (anomaly/outlier)
            normalized_scores: 0.0 (very nominal) to 1.0 (highly anomalous)
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            raise RuntimeError("Anomaly detector is not trained.")

        feat_df = extract_cut_level_features(df)
        missing = [c for c in self.feature_names if c not in feat_df.columns]
        for m in missing:
            feat_df[m] = self.baseline_stats.get(m, {}).get("mean", 0.0)

        X = feat_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # score_samples returns opposite of anomaly score (lower is more anomalous)
        raw_scores = -self.model.score_samples(X_scaled)
        min_s = float(np.min(raw_scores))
        max_s = float(np.max(raw_scores))
        if max_s > min_s:
            norm_scores = (raw_scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(raw_scores)

        preds = self.model.predict(X_scaled)
        return preds, norm_scores

    def evaluate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single telemetry point for real-time dashboard health monitoring.
        Classifies strictly into: NORMAL, WARNING, or ANOMALY.
        Computes detailed sensor contributions explaining which sensors caused the anomaly.
        """
        # Sensor process values
        f_res = float(record.get("force_resultant", 150.0))
        vib_rms = float(record.get("vibration_rms", 1.0))
        ae_rms = float(record.get("ae_rms", 0.25))
        sp_rpm = float(record.get("spindle_speed_rpm", NOMINAL_SPINDLE_SPEED_RPM))
        feed_mm = float(record.get("feed_rate_mm_min", NOMINAL_FEED_RATE_MM_MIN))
        doc_mm = float(record.get("depth_of_cut_mm", NOMINAL_DEPTH_OF_CUT_MM))

        # Check Physical Rule Boundaries
        critical_reasons = []
        warning_reasons = []

        if f_res >= LIMITS["force_resultant_critical"]:
            critical_reasons.append(f"Resultant Force critical: {f_res:.1f} N (limit {LIMITS['force_resultant_critical']} N)")
        elif f_res >= LIMITS["force_resultant_warning"]:
            warning_reasons.append(f"High Cutting Force: {f_res:.1f} N")

        if vib_rms >= LIMITS["vibration_rms_critical"]:
            critical_reasons.append(f"Severe Vibration Chatter: {vib_rms:.2f} g (limit {LIMITS['vibration_rms_critical']} g)")
        elif vib_rms >= LIMITS["vibration_rms_warning"]:
            warning_reasons.append(f"Elevated Vibration: {vib_rms:.2f} g")

        if ae_rms >= LIMITS["ae_rms_critical"]:
            critical_reasons.append(f"Acoustic Emission spike: {ae_rms:.3f} V (edge micro-fracture)")
        elif ae_rms >= LIMITS["ae_rms_warning"]:
            warning_reasons.append(f"High Acoustic Emission: {ae_rms:.3f} V")

        # Isolation Forest Score
        ml_score = 0.20
        is_forest_outlier = False
        if self.is_trained and self.model is not None and self.scaler is not None:
            try:
                row_df = pd.DataFrame([record])
                preds, scores = self.score_samples(row_df)
                ml_score = float(scores[0])
                is_forest_outlier = (preds[0] == -1)
            except Exception:
                ml_score = 0.30

        is_ground_anomaly = (int(record.get("is_anomaly", 0)) == 1)

        # Classification into strictly NORMAL, WARNING, ANOMALY
        if critical_reasons or is_ground_anomaly or is_forest_outlier or ml_score >= 0.72:
            status = STATUS_ANOMALY
            if critical_reasons:
                reason = " | ".join(critical_reasons)
            elif is_ground_anomaly and record.get("anomaly_type") not in ("NORMAL", None):
                reason = str(record.get("anomaly_type"))
            else:
                reason = f"IsolationForest Outlier (Score: {ml_score:.2f}) - Dynamic sensor irregularity"
        elif warning_reasons or ml_score >= 0.48 or f_res >= 340.0 or vib_rms >= 2.8:
            status = STATUS_WARNING
            reason = " | ".join(warning_reasons) if warning_reasons else f"Elevated Sensor Signatures (Score: {ml_score:.2f}) - Minor irregularity"
        else:
            status = STATUS_NORMAL
            reason = "Process signals within nominal parameters"

        # Calculate Sensor Contribution Breakdown (Explainability)
        sensor_contributions = self._compute_sensor_contributions(record)

        return {
            "status": status,
            "anomaly_score": round(ml_score, 3),
            "reason": reason,
            "sensor_contributions": sensor_contributions,
            "force_resultant": f_res,
            "vibration_rms": vib_rms,
            "ae_rms": ae_rms,
            "spindle_speed_rpm": sp_rpm,
            "feed_rate_mm_min": feed_mm,
            "depth_of_cut_mm": doc_mm
        }

    def _compute_sensor_contributions(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compute relative contribution of each sensor value to the abnormal condition.
        Uses standardized z-score deviations and physical boundary excesses.
        """
        contributions = []
        for feat in self.feature_names:
            val = float(record.get(feat, 0.0))
            meta = SENSOR_METADATA.get(feat, {"label": feat, "unit": "", "nominal": val, "warn_thresh": val * 1.5, "crit_thresh": val * 2.0})

            stats = self.baseline_stats.get(feat, {"mean": meta["nominal"], "std": 1.0})
            mean_v = stats["mean"]
            std_v = max(1e-4, stats["std"])

            # Standardized z-score deviation
            z_score = abs(val - mean_v) / std_v
            # Percentage deviation from nominal
            dev_pct = ((val - mean_v) / max(1e-4, abs(mean_v))) * 100.0

            # Physical limit ratio
            crit_ratio = val / max(1e-4, meta["crit_thresh"])

            # Normalized contribution weight
            impact_score = max(0.0, z_score * 0.5 + max(0.0, crit_ratio - 0.7) * 2.0)

            # Assign severity for this specific sensor
            if val >= meta["crit_thresh"] or z_score >= 3.0:
                severity = "ANOMALY"
            elif val >= meta["warn_thresh"] or z_score >= 2.0:
                severity = "WARNING"
            else:
                severity = "NORMAL"

            contributions.append({
                "feature": feat,
                "label": meta["label"],
                "value": round(val, 3),
                "nominal": round(mean_v, 3),
                "unit": meta["unit"],
                "deviation_pct": round(dev_pct, 1),
                "z_score": round(z_score, 2),
                "impact_score": round(impact_score, 2),
                "severity": severity,
                "warn_thresh": meta["warn_thresh"],
                "crit_thresh": meta["crit_thresh"]
            })

        # Sort descending by impact score
        contributions.sort(key=lambda x: x["impact_score"], reverse=True)

        # Normalize relative contribution percentages to sum to 100%
        total_impact = sum(c["impact_score"] for c in contributions)
        for c in contributions:
            if total_impact > 0:
                c["relative_contribution_pct"] = round((c["impact_score"] / total_impact) * 100.0, 1)
            else:
                c["relative_contribution_pct"] = round(100.0 / len(contributions), 1)

        return contributions

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fast vectorized classification of all rows into NORMAL, WARNING, or ANOMALY.
        Returns an enriched DataFrame with classification labels, scores, and primary factors.
        """
        res_df = df.copy()
        if not self.is_trained:
            self.train(df)

        preds, scores = self.score_samples(res_df)
        res_df["ml_anomaly_score"] = np.round(scores, 3)

        # Vectorized classification
        is_ground = res_df.get("is_anomaly", pd.Series(0, index=res_df.index)) == 1
        is_crit_f = res_df.get("force_resultant", pd.Series(0, index=res_df.index)) >= LIMITS["force_resultant_critical"]
        is_crit_v = res_df.get("vibration_rms", pd.Series(0, index=res_df.index)) >= LIMITS["vibration_rms_critical"]
        is_crit_ae = res_df.get("ae_rms", pd.Series(0, index=res_df.index)) >= LIMITS["ae_rms_critical"]
        
        is_anom = (preds == -1) | (scores >= 0.72) | is_ground | is_crit_f | is_crit_v | is_crit_ae
        
        is_warn_f = res_df.get("force_resultant", pd.Series(0, index=res_df.index)) >= LIMITS["force_resultant_warning"]
        is_warn_v = res_df.get("vibration_rms", pd.Series(0, index=res_df.index)) >= LIMITS["vibration_rms_warning"]
        is_warn_ae = res_df.get("ae_rms", pd.Series(0, index=res_df.index)) >= LIMITS["ae_rms_warning"]
        is_warn = (~is_anom) & ((scores >= 0.48) | is_warn_f | is_warn_v | is_warn_ae)
        
        classification = np.where(is_anom, STATUS_ANOMALY, np.where(is_warn, STATUS_WARNING, STATUS_NORMAL))
        res_df["classification"] = classification

        # Determine primary contributing factor vectorized
        top_factors = []
        for i, row in res_df.iterrows():
            if row["classification"] == STATUS_NORMAL:
                top_factors.append("Nominal Operation")
            elif is_ground.iloc[i] and str(row.get("anomaly_type", "")) not in ("NORMAL", "nan", ""):
                top_factors.append(str(row["anomaly_type"]))
            elif row.get("force_resultant", 0) >= LIMITS["force_resultant_warning"]:
                top_factors.append("Cutting Force Spike")
            elif row.get("vibration_rms", 0) >= LIMITS["vibration_rms_warning"]:
                top_factors.append("Chatter Vibration")
            elif row.get("ae_rms", 0) >= LIMITS["ae_rms_warning"]:
                top_factors.append("Acoustic Emission Spike")
            else:
                top_factors.append("Multivariate Outlier")
                
        res_df["top_contributing_sensor"] = top_factors
        return res_df

    def save(self, filepath: Optional[Path] = None):
        """Save anomaly model to disk."""
        target_path = filepath or ANOMALY_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "baseline_stats": self.baseline_stats,
            "contamination": self.contamination
        }, str(target_path))

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Load anomaly model from disk."""
        target_path = filepath or ANOMALY_MODEL_PATH
        if not target_path.exists():
            return False
        try:
            data = joblib.load(str(target_path))
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data.get("feature_names", ANOMALY_FEATURE_COLS)
            self.baseline_stats = data.get("baseline_stats", {})
            self.contamination = data.get("contamination", 0.05)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading anomaly detector from {target_path}: {e}")
            return False


def get_trained_anomaly_detector(sample_df: Optional[pd.DataFrame] = None) -> CNCAnomalyDetector:
    """Helper to get an active trained CNCAnomalyDetector instance."""
    detector = CNCAnomalyDetector()
    if detector.load() and len(detector.baseline_stats) > 0 and len(detector.feature_names) >= 10:
        return detector

    if sample_df is not None:
        detector.train(sample_df)
        detector.save()
        return detector

    raise RuntimeError("No saved anomaly model found and no sample data provided.")
