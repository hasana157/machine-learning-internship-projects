"""
Feature engineering for the energy forecasting pipeline.

Adds calendar features, lag features, and rolling statistics.
All lag/rolling windows are read from config.yaml, so tuning the
lookback horizon never requires touching this file.
"""

import pandas as pd


def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    feat_cfg = config["features"]

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    for lag in feat_cfg["lag_steps"]:
        df[f"lag_{lag}"] = df["consumption"].shift(lag)

    for window in feat_cfg["rolling_windows"]:
        df[f"roll_mean_{window}"] = df["consumption"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df["consumption"].shift(1).rolling(window).std()

    return df.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("date", "consumption")]
