"""Feature engineering for the sales forecasting model.

All rolling/lag features are shifted so that no row ever sees
information from its own day or the future — this is what makes
the evaluation honest for a time series problem.
"""

import pandas as pd


def create_features(df: pd.DataFrame, lags=(1, 7, 14, 28), windows=(7, 14, 30)) -> pd.DataFrame:
    """Return a copy of df with lag, rolling-stat, and calendar features added.

    Expects columns: date (datetime64), sales (float). Optional columns
    is_promo / is_holiday are passed through unchanged if present.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for lag in lags:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for w in windows:
        shifted = df["sales"].shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}"] = shifted.rolling(w).std()

    # Calendar features — cheap, high signal, always known ahead of time
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    return df.dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Every model input column — everything except date/sales/target."""
    exclude = {"date", "sales"}
    return [c for c in df.columns if c not in exclude]
