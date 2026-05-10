"""
Feature engineering module for SentinelFlow.
Transforms raw time-series sensor data into ML-ready features.
"""

import pandas as pd
import numpy as np
from typing import List

from src.utils import setup_logger

logger = setup_logger(__name__)

# Track the feature names added by engineer_features
_FEATURE_NAMES = []

def engineer_features(df: pd.DataFrame, window: int = 10, lag_steps: List[int] = None) -> pd.DataFrame:
    """
    Engineer time-series features from raw sensor readings.

    Args:
        df (pd.DataFrame): Raw sensor DataFrame.
        window (int): Rolling window size.
        lag_steps (List[int]): Steps to look back for lag features.

    Returns:
        pd.DataFrame: DataFrame with newly engineered features (and dropped NaNs).
    """
    if lag_steps is None:
        lag_steps = [1, 3, 6]
        
    df_feat = df.copy()
    sensors = ["temp", "vibration", "pressure", "current"]
    
    global _FEATURE_NAMES
    _FEATURE_NAMES = []
    
    original_len = len(df_feat)
    
    for sensor in sensors:
        # Rolling statistics
        df_feat[f"{sensor}_rolling_mean"] = df_feat[sensor].rolling(window=window).mean()
        df_feat[f"{sensor}_rolling_std"] = df_feat[sensor].rolling(window=window).std().fillna(0)
        df_feat[f"{sensor}_rolling_min"] = df_feat[sensor].rolling(window=window).min()
        df_feat[f"{sensor}_rolling_max"] = df_feat[sensor].rolling(window=window).max()
        
        _FEATURE_NAMES.extend([
            f"{sensor}_rolling_mean", f"{sensor}_rolling_std",
            f"{sensor}_rolling_min", f"{sensor}_rolling_max"
        ])
        
        # Lag features
        for lag in lag_steps:
            col_name = f"{sensor}_lag_{lag}"
            df_feat[col_name] = df_feat[sensor].shift(lag)
            _FEATURE_NAMES.append(col_name)
            
        # Rate of change
        df_feat[f"{sensor}_diff_1"] = df_feat[sensor].diff(1)
        df_feat[f"{sensor}_diff_2"] = df_feat[sensor].diff(2)
        _FEATURE_NAMES.extend([f"{sensor}_diff_1", f"{sensor}_diff_2"])
        
    # Cross-sensor features
    df_feat["temp_vibration_ratio"] = df_feat["temp"] / (df_feat["vibration"] + 1e-6)
    df_feat["pressure_current_product"] = df_feat["pressure"] * df_feat["current"]
    
    _FEATURE_NAMES.extend(["temp_vibration_ratio", "pressure_current_product"])
    
    # Calculate Z-Scores based on running statistics up to that point
    # To simulate real-time, we use expanding window instead of full dataset stats
    for sensor in sensors:
        expanding_mean = df_feat[sensor].expanding().mean()
        expanding_std = df_feat[sensor].expanding().std().replace(0, 1e-6).fillna(1e-6)
        df_feat[f"z_{sensor}"] = (df_feat[sensor] - expanding_mean) / expanding_std
        # Z-scores are also features
        _FEATURE_NAMES.append(f"z_{sensor}")

    # Drop NaNs introduced by rolling and lag operations
    df_feat = df_feat.dropna()
    dropped_rows = original_len - len(df_feat)
    
    logger.info(f"Engineered {len(_FEATURE_NAMES)} features.")
    logger.info(f"Dropped {dropped_rows} rows containing NaNs from feature engineering.")
    
    # Add raw sensors as features as well
    _FEATURE_NAMES = sensors + _FEATURE_NAMES
    
    return df_feat

def get_feature_names() -> List[str]:
    """
    Retrieve the list of engineered feature names.

    Returns:
        List[str]: List of feature names.
    """
    return _FEATURE_NAMES
