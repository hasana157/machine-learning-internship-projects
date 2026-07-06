"""
Feature engineering pipeline for demand forecasting.

Generates 28+ features including:
    - Calendar features (day, week, month, holidays, etc.)
    - Lag features (1, 7, 14, 28, 365 days)
    - Rolling statistics (mean, std, min, max, median, EWM)
    - Velocity and acceleration features
    - Promotional features
    - Store metadata (categorical and numeric)

All lag and rolling features are computed per store_id group to avoid data leakage.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from dateutil.easter import easter

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline.

    Process:
        1. Parse and validate date column
        2. Generate calendar features
        3. Compute lag features (per store group)
        4. Compute rolling statistics (per store group)
        5. Compute velocity/acceleration features
        6. Add promotional lag features
        7. Drop NaN rows
        8. Log summary

    Args:
        df: Input DataFrame with at minimum [store_id, date, sales].
        config: Configuration dictionary.

    Returns:
        Feature-engineered DataFrame with all computed features.
    """
    df = df.copy()
    logger.info("🔧 Starting feature engineering pipeline")

    initial_rows = len(df)

    # Ensure date is datetime
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"])

    # Calendar features
    df = _add_calendar_features(df, config)

    # Lag features (per store group)
    lag_steps = config["features"]["lag_steps"]
    for lag in lag_steps:
        df[f"lag_{lag}"] = df.groupby("store_id")["sales"].shift(lag)

    # Rolling statistics (computed on shift(1) to avoid leakage)
    rolling_windows = config["features"]["rolling_windows"]
    for window in rolling_windows:
        df[f"roll_mean_{window}"] = (
            df.groupby("store_id")["sales"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"roll_std_{window}"] = (
            df.groupby("store_id")["sales"].shift(1).rolling(window=window, min_periods=1).std()
        )

    df[f"roll_min_7"] = (
        df.groupby("store_id")["sales"].shift(1).rolling(window=7, min_periods=1).min()
    )
    df[f"roll_max_7"] = (
        df.groupby("store_id")["sales"].shift(1).rolling(window=7, min_periods=1).max()
    )
    df[f"roll_median_14"] = (
        df.groupby("store_id")["sales"].shift(1).rolling(window=14, min_periods=1).median()
    )

    # EWM features
    ewm_spans = config["features"]["ewm_spans"]
    for span in ewm_spans:
        df[f"roll_ewm_{span}"] = (
            df.groupby("store_id")["sales"].shift(1).ewm(span=span, adjust=False).mean()
        )

    # Velocity and acceleration
    df["sales_velocity_7"] = df["lag_1"] - df["lag_7"]
    df["sales_accel"] = df["lag_1"] - 2 * df["lag_7"] + df["lag_14"]

    # Coefficient of variation
    df["cv_7"] = df["roll_std_7"] / (df["roll_mean_7"] + 1e-6)

    # Promo features
    df["promo_lag_1"] = df.groupby("store_id")["promo"].shift(1)
    df["promo_roll_7"] = (
        df.groupby("store_id")["promo"].shift(1).rolling(window=7, min_periods=1).sum()
    )

    # Drop NaN rows
    rows_before_dropna = len(df)
    df = df.dropna()
    rows_after_dropna = len(df)

    logger.info(
        f"✂️  Dropped NaN rows: {rows_before_dropna:,} → {rows_after_dropna:,} "
        f"({rows_before_dropna - rows_after_dropna:,} removed)"
    )
    logger.info(f"📊 Total features engineered: {len(get_numeric_feature_cols()) + len(get_categorical_feature_cols())}")

    return df


def _add_calendar_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add calendar-based features from date column.

    Args:
        df: DataFrame with date column.
        config: Configuration dictionary.

    Returns:
        DataFrame with added calendar features.
    """
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Days to nearest Christmas
    christmas_day_cap = config["features"]["days_to_christmas_cap"]
    df["days_to_christmas"] = df["date"].apply(
        lambda d: _days_to_christmas(d, christmas_day_cap)
    )

    # Days to nearest Easter
    easter_day_cap = config["features"]["days_to_easter_cap"]
    df["days_to_easter"] = df["date"].apply(
        lambda d: _days_to_easter(d, easter_day_cap)
    )

    return df


def _days_to_christmas(date: pd.Timestamp, cap: int) -> int:
    """
    Calculate days to nearest Christmas (Dec 25), capped at specified value.

    Args:
        date: Date to calculate from.
        cap: Maximum absolute value to cap at.

    Returns:
        Capped days difference.
    """
    year = date.year
    christmas = pd.Timestamp(year=year, month=12, day=25)

    if date > christmas:
        christmas = pd.Timestamp(year=year + 1, month=12, day=25)

    days_diff = abs((date - christmas).days)
    return min(days_diff, cap)


def _days_to_easter(date: pd.Timestamp, cap: int) -> int:
    """
    Calculate days to nearest Easter Sunday, capped at specified value.

    Args:
        date: Date to calculate from.
        cap: Maximum absolute value to cap at.

    Returns:
        Capped days difference.
    """
    year = date.year
    easter_date = easter(year)
    easter_ts = pd.Timestamp(easter_date)

    if date > easter_ts:
        easter_ts = pd.Timestamp(easter(year + 1))

    days_diff = abs((date - easter_ts).days)
    return min(days_diff, cap)


def get_numeric_feature_cols() -> List[str]:
    """
    Return list of numeric feature column names.

    Includes: calendar features, all lag/rolling features, velocity, CV, promo features.

    Returns:
        List of numeric feature column names.
    """
    return [
        # Calendar
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "days_to_christmas",
        "days_to_easter",
        # Lag features
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "lag_365",
        # Rolling mean
        "roll_mean_7",
        "roll_mean_14",
        "roll_mean_28",
        # Rolling std
        "roll_std_7",
        "roll_std_28",
        # Rolling min/max/median
        "roll_min_7",
        "roll_max_7",
        "roll_median_14",
        # EWM
        "roll_ewm_7",
        "roll_ewm_28",
        # Velocity and acceleration
        "sales_velocity_7",
        "sales_accel",
        "cv_7",
        # Store metadata (numeric)
        "store_id",
        "customers",
        "competition_distance",
        # Promo features
        "promo",
        "promo_lag_1",
        "promo_roll_7",
    ]


def get_categorical_feature_cols() -> List[str]:
    """
    Return list of categorical feature column names.

    Categorical features are one-hot encoded during modeling.

    Returns:
        List of categorical feature column names.
    """
    return [
        "store_type",
        "assortment",
        "state_holiday",
    ]


def get_target_col() -> str:
    """
    Return the target column name.

    Returns:
        "sales"
    """
    return "sales"
