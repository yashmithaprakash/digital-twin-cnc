"""
Machine Learning Tool Wear Prediction Model (Scikit-Learn).
Implements regression pipeline for predicting cutting tool flank wear (VB in µm),
calculating performance metrics (MAE, RMSE, R²), and evaluating feature importances.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.config import WEAR_MODEL_PATH, WEAR_MAX_THRESHOLD_UM
from src.feature_engineering import CORE_MODEL_FEATURES, extract_cut_level_features


class ToolWearModel:
    """
    Scikit-learn based Regression Model for CNC Flank Wear (VB) Estimation.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = CORE_MODEL_FEATURES
        self.metrics: Dict[str, float] = {}
        self.is_trained: bool = False
        self.feature_importances_: Dict[str, float] = {}

    def _build_estimator(self):
        if self.model_type == "random_forest":
            regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            regressor = GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", regressor)
        ])

    def train(self, df: pd.DataFrame,
              target_col: str = "flank_wear_um",
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the model pipeline and calculate MAE, RMSE, and R².
        """
        # Ensure engineered features exist
        feat_df = extract_cut_level_features(df)

        # Select available features from CORE_MODEL_FEATURES
        available_features = [col for col in self.feature_names if col in feat_df.columns]
        if len(available_features) < 3:
            raise ValueError("Insufficient feature columns in dataset for training.")
        self.feature_names = available_features

        X = feat_df[self.feature_names].values
        y = feat_df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )

        self._build_estimator()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Predictions on test set
        y_pred = self.pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        self.metrics = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # Extract feature importances
        regressor = self.pipeline.named_steps["regressor"]
        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            self.feature_importances_ = {
                feat: round(float(imp), 4)
                for feat, imp in zip(self.feature_names, importances)
            }
            # Sort descending
            self.feature_importances_ = dict(
                sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True)
            )

        return self.metrics

    def predict(self, df_or_features: pd.DataFrame) -> np.ndarray:
        """
        Predict flank wear (VB in µm) for given inputs.
        """
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model is not trained. Call train() or load() first.")

        if isinstance(df_or_features, pd.DataFrame):
            feat_df = extract_cut_level_features(df_or_features)
            missing = [c for c in self.feature_names if c not in feat_df.columns]
            for m in missing:
                feat_df[m] = 0.0
            X = feat_df[self.feature_names].values
        else:
            X = np.asarray(df_or_features)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        preds = self.pipeline.predict(X)
        # Wear cannot physically be negative
        return np.maximum(0.0, preds)

    def predict_health_pct(self, predicted_wear_um: float) -> float:
        """
        Convert predicted flank wear (µm) to a standardized 0-100% Tool Health score.
        """
        pct = 100.0 * (1.0 - (predicted_wear_um / WEAR_MAX_THRESHOLD_UM))
        return float(np.clip(pct, 0.0, 100.0))

    def save(self, filepath: Optional[Path] = None):
        """Save fitted model artifact to disk safely."""
        target_path = filepath or WEAR_MODEL_PATH
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "feature_importances": self.feature_importances_,
                "model_type": self.model_type
            }, str(target_path))
        except Exception as e:
            print(f"Notice: Model saving skipped ({e})")

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Load model artifact from disk."""
        target_path = filepath or WEAR_MODEL_PATH
        if not target_path.exists():
            return False
        try:
            data = joblib.load(str(target_path))
            self.pipeline = data.get("pipeline")
            self.feature_names = data.get("feature_names", [])
            self.metrics = data.get("metrics", {})
            self.feature_importances_ = data.get("feature_importances", {})
            self.model_type = data.get("model_type", "random_forest")
            if self.pipeline is not None and len(self.feature_names) >= 3 and len(self.metrics) > 0:
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model from {target_path}: {e}")
            return False


def get_trained_tool_wear_model(sample_df: Optional[pd.DataFrame] = None) -> ToolWearModel:
    """
    Helper to get an active trained ToolWearModel instance.
    Loads from disk if available, otherwise trains on sample_df.
    """
    model = ToolWearModel()
    if model.load() and model.is_trained and len(model.metrics) > 0:
        return model

    if sample_df is not None:
        try:
            model.train(sample_df)
            model.save()
            return model
        except Exception as e:
            print(f"Notice: Failed to train wear model on sample_df: {e}")

    raise RuntimeError("No valid saved wear model found and dataset training failed.")
