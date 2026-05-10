"""
Inference module for SentinelFlow.
Handles real-time scoring and prediction on new sensor data.
"""

import pandas as pd
from typing import Tuple

from src.utils import setup_logger
from src.model import AnomalyDetector
from src.features import engineer_features

logger = setup_logger(__name__)

def run_detection(df_raw: pd.DataFrame, detector: AnomalyDetector, config: dict) -> pd.DataFrame:
    """
    Run anomaly detection on raw sensor data.

    Args:
        df_raw (pd.DataFrame): Raw sensor data containing [timestamp, temp, vibration, pressure, current].
        detector (AnomalyDetector): Loaded trained model.
        config (dict): Project configuration.

    Returns:
        pd.DataFrame: DataFrame with predictions and anomaly scores.
    """
    logger.info(f"Running detection on {len(df_raw)} records...")
    
    # 1. Engineer features
    window = config["features"]["rolling_window"]
    lag_steps = config["features"]["lag_steps"]
    
    # Ensure dataframe is sorted by time if there's a timestamp
    if "timestamp" in df_raw.columns:
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
        
    df_feat = engineer_features(df_raw, window=window, lag_steps=lag_steps)
    
    # 2. Extract feature columns required by model
    X = df_feat[detector.feature_names]
    
    # 3. Predict & Score
    predictions = detector.predict(X)
    scores = detector.score_samples(X)
    z_predictions = detector.predict_zscore(X)
    
    # 4. Attach results
    df_results = df_feat.copy()
    df_results["predicted_label"] = predictions
    df_results["anomaly_score"] = scores
    df_results["zscore_label"] = z_predictions
    
    anomalies_found = int(predictions.sum())
    logger.info(f"Detection complete. Found {anomalies_found} anomalies.")
    
    return df_results
