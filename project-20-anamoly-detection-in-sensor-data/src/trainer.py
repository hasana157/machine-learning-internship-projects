"""
Training pipeline module for SentinelFlow.
Handles the execution of data generation, feature engineering, and model training.
"""

import time
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

from src.utils import setup_logger
from src.data_generator import generate_sensor_data
from src.features import engineer_features, get_feature_names
from src.model import AnomalyDetector

logger = setup_logger(__name__)

def run_training_pipeline(config: Dict[str, Any]) -> Tuple[AnomalyDetector, float]:
    """
    Execute the full training pipeline.

    Args:
        config (Dict[str, Any]): Project configuration.

    Returns:
        Tuple[AnomalyDetector, float]: Trained model and time taken in seconds.
    """
    start_time = time.time()
    
    # 1. Generate data
    logger.info("=== Starting Data Generation ===")
    df, anomaly_labels = generate_sensor_data(config)
    
    # Save raw data
    data_path = Path(config["paths"]["data"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    
    # 2. Engineer features
    logger.info("=== Starting Feature Engineering ===")
    window = config["features"]["rolling_window"]
    lag_steps = config["features"]["lag_steps"]
    df_feat = engineer_features(df, window=window, lag_steps=lag_steps)
    
    # 3. Train/Eval Split
    logger.info("=== Starting Model Training ===")
    train_split = config["evaluation"]["train_split"]
    split_idx = int(len(df_feat) * train_split)
    
    # We train only on the first portion
    df_train = df_feat.iloc[:split_idx]
    
    # IMPORTANT: Isolation forest is unsupervised, but to ensure a clean baseline, 
    # we filter out known anomalies from the training set if possible, 
    # or just train on the raw stream (which includes 3% anomalies).
    # The prompt says: "first 80% for training (normal points only)"
    df_train_normal = df_train[df_train["is_anomaly"] == 0].copy()
    
    # Extract features for training (exclude timestamp, is_anomaly)
    feature_cols = get_feature_names()
    X_train = df_train_normal[feature_cols]
    
    # 4. Train Model
    detector = AnomalyDetector(config)
    detector.fit(X_train)
    
    # 5. Save Artifacts
    logger.info("=== Saving Artifacts ===")
    model_path = Path(config["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(model_path))
    
    scaler_path = Path(config["paths"]["scaler"])
    joblib.dump(detector.scaler, str(scaler_path))
    logger.info(f"Scaler saved to {scaler_path}")
    
    fn_path = Path(config["paths"]["feature_names"])
    with open(fn_path, "w") as f:
        json.dump(detector.feature_names, f)
    logger.info(f"Feature names saved to {fn_path}")
    
    time_taken = time.time() - start_time
    
    # Log summary
    logger.info(f"Training completed in {time_taken:.2f} seconds.")
    logger.info(f"Training set size (normal points only): {len(X_train)}")
    logger.info(f"Contamination rate config: {detector.contamination}")
    logger.info(f"Threshold value calculated: {detector.threshold:.4f}")
    
    return detector, time_taken
