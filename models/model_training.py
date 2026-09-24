"""
Offline Model Training Script for CNC Milling Digital Twin.
Trains both the Scikit-learn Tool Wear Regression Model (RandomForest)
and the Isolation Forest Anomaly Detection Model, saving model artifacts.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_dataset, init_db
from src.tool_wear_model import ToolWearModel
from src.anomaly_detection import CNCAnomalyDetector
from utils.config import WEAR_MODEL_PATH, ANOMALY_MODEL_PATH


def train_all_models(force_retrain: bool = True):
    print("=" * 65)
    print("  CNC DIGITAL TWIN (PRJ_422) - ML MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Initialize SQLite database
    print("\n[1/4] Initializing local SQLite database...")
    init_db()
    print("      Database verified at data/digital_twin_cnc.db")

    # 2. Load dataset
    print("\n[2/4] Loading CNC dataset...")
    df, is_real, source_label = load_dataset()
    print(f"      Source: {source_label}")
    print(f"      Total milling cuts: {len(df)}")
    print(f"      Wear range: {df['flank_wear_um'].min():.1f} µm -> {df['flank_wear_um'].max():.1f} µm")

    # 3. Train Tool Wear Regression Model
    print("\n[3/4] Training Scikit-learn Tool Wear Regression Model (RandomForest)...")
    wear_model = ToolWearModel(model_type="random_forest")
    metrics = wear_model.train(df)
    wear_model.save(WEAR_MODEL_PATH)

    print("      Model training complete!")
    print(f"      - Mean Absolute Error (MAE): {metrics['MAE']:.3f} µm")
    print(f"      - Root Mean Squared Error (RMSE): {metrics['RMSE']:.3f} µm")
    print(f"      - Coefficient of Determination (R²): {metrics['R2']:.4f}")
    print(f"      - Train/Test Samples: {metrics['train_samples']} / {metrics['test_samples']}")
    print(f"      - Artifact saved to: {WEAR_MODEL_PATH}")

    print("\n      Top 5 Most Important Sensor Features:")
    for feat, imp in list(wear_model.feature_importances_.items())[:5]:
        print(f"        * {feat:30s}: {imp * 100:.2f}%")

    # 4. Train Anomaly Detection Model (Isolation Forest)
    print("\n[4/4] Training Isolation Forest Anomaly Detection Model...")
    anomaly_model = CNCAnomalyDetector(contamination=0.05)
    anomaly_model.train(df)
    anomaly_model.save(ANOMALY_MODEL_PATH)
    print(f"      Anomaly model saved to: {ANOMALY_MODEL_PATH}")

    print("\n" + "=" * 65)
    print("  SUCCESS: All CNC Digital Twin ML models trained and saved!")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    train_all_models()
