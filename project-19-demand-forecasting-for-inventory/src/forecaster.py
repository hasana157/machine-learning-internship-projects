"""
Future demand forecasting with scenario simulation.

Provides:
    - Recursive multi-step forecasting (auto-regressive)
    - Promotion scenario simulation (no promo, weekly, aggressive)
    - Confidence intervals (±12% by default)
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from features import engineer_features, get_numeric_feature_cols, get_categorical_feature_cols, get_target_col
from model import DemandForecaster

logger = logging.getLogger(__name__)


def forecast_future(
    model: DemandForecaster,
    df: pd.DataFrame,
    store_id: int,
    horizon_days: int,
    promo_schedule: List[int],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate recursive multi-step forecast for a specific store.

    Algorithm:
        1. Filter to store_id, take last 400 rows as context
        2. For each future day:
           a. Build feature vector using context history
           b. Predict next day's sales
           c. Add to context for next iteration (auto-regressive)
           d. Clip prediction to non-negative

    Args:
        model: Trained DemandForecaster.
        df: Full historical DataFrame (with engineered features).
        store_id: Store ID to forecast.
        horizon_days: Number of days to forecast ahead.
        promo_schedule: List of 0/1 indicating promo on each future day.
        config: Configuration dictionary.

    Returns:
        DataFrame with columns:
            [date, store_id, forecasted_sales, lower_bound, upper_bound, promo]
    """
    # Filter to store and get last 400 rows as context
    store_df = df[df["store_id"] == store_id].copy()
    if len(store_df) == 0:
        logger.warning(f"Store {store_id} not found in data")
        return pd.DataFrame()

    context = store_df.tail(400).copy()
    context = context.sort_values("date").reset_index(drop=True)

    if len(context) == 0:
        logger.warning(f"Store {store_id} has no history after filtering")
        return pd.DataFrame()

    last_date = context["date"].max()
    store_type = context["store_type"].iloc[-1]
    assortment = context["assortment"].iloc[-1]
    competition_distance = context["competition_distance"].iloc[-1]

    forecasts = []

    for day_idx in range(horizon_days):
        next_date = last_date + pd.Timedelta(days=day_idx + 1)

        # Extract feature values from context
        feature_dict = {
            "store_id": store_id,
            "date": next_date,
            "sales": 0,  # Will be updated
            "customers": context["customers"].iloc[-1],
            "open": 1,
            "promo": promo_schedule[day_idx] if day_idx < len(promo_schedule) else 0,
            "state_holiday": 0,
            "school_holiday": 1 if next_date.month in [7, 8] else 0,
            "store_type": store_type,
            "assortment": assortment,
            "competition_distance": competition_distance,
            "competition_open_since_month": context["competition_open_since_month"].iloc[-1]
            if "competition_open_since_month" in context.columns
            else 1,
            "competition_open_since_year": context["competition_open_since_year"].iloc[-1]
            if "competition_open_since_year" in context.columns
            else 2000,
            "promo2": context["promo2"].iloc[-1] if "promo2" in context.columns else 0,
            "promo2_since_week": context["promo2_since_week"].iloc[-1]
            if "promo2_since_week" in context.columns
            else 1,
            "promo2_since_year": context["promo2_since_year"].iloc[-1]
            if "promo2_since_year" in context.columns
            else 2000,
            "promo_interval": context["promo_interval"].iloc[-1]
            if "promo_interval" in context.columns
            else None,
        }

        # Add lags and rolling features from context
        feature_dict["lag_1"] = context["sales"].iloc[-1]
        feature_dict["lag_7"] = context["sales"].iloc[-7] if len(context) >= 7 else context["sales"].iloc[0]
        feature_dict["lag_14"] = context["sales"].iloc[-14] if len(context) >= 14 else context["sales"].iloc[0]
        feature_dict["lag_28"] = context["sales"].iloc[-28] if len(context) >= 28 else context["sales"].iloc[0]
        feature_dict["lag_365"] = (
            context["sales"].iloc[-365] if len(context) >= 365 else context["sales"].mean()
        )

        feature_dict["roll_mean_7"] = context["sales"].tail(7).mean()
        feature_dict["roll_mean_14"] = context["sales"].tail(14).mean()
        feature_dict["roll_mean_28"] = context["sales"].tail(28).mean()

        feature_dict["roll_std_7"] = context["sales"].tail(7).std()
        feature_dict["roll_std_28"] = context["sales"].tail(28).std()

        feature_dict["roll_min_7"] = context["sales"].tail(7).min()
        feature_dict["roll_max_7"] = context["sales"].tail(7).max()
        feature_dict["roll_median_14"] = context["sales"].tail(14).median()

        feature_dict["roll_ewm_7"] = (
            context["sales"].tail(7).ewm(span=7, adjust=False).mean().iloc[-1]
        )
        feature_dict["roll_ewm_28"] = (
            context["sales"].tail(28).ewm(span=28, adjust=False).mean().iloc[-1]
        )

        feature_dict["sales_velocity_7"] = feature_dict["lag_1"] - feature_dict["lag_7"]
        feature_dict["sales_accel"] = (
            feature_dict["lag_1"] - 2 * feature_dict["lag_7"] + feature_dict["lag_14"]
        )

        feature_dict["cv_7"] = (
            feature_dict["roll_std_7"] / (feature_dict["roll_mean_7"] + 1e-6)
        )

        feature_dict["promo_lag_1"] = context["promo"].iloc[-1]
        feature_dict["promo_roll_7"] = context["promo"].tail(7).sum()

        # Add calendar features
        feature_dict["day_of_week"] = next_date.dayofweek
        feature_dict["day_of_month"] = next_date.day
        feature_dict["week_of_year"] = next_date.isocalendar()[1]
        feature_dict["month"] = next_date.month
        feature_dict["quarter"] = (next_date.month - 1) // 3 + 1
        feature_dict["year"] = next_date.year
        feature_dict["is_weekend"] = 1 if next_date.dayofweek >= 5 else 0
        feature_dict["is_month_start"] = 1 if next_date.day == 1 else 0
        feature_dict["is_month_end"] = 1 if (next_date + pd.Timedelta(days=1)).day == 1 else 0

        from dateutil.easter import easter

        christmas_day_cap = config["features"]["days_to_christmas_cap"]
        christmas = pd.Timestamp(year=next_date.year, month=12, day=25)
        if next_date > christmas:
            christmas = pd.Timestamp(year=next_date.year + 1, month=12, day=25)
        feature_dict["days_to_christmas"] = min(
            abs((next_date - christmas).days), christmas_day_cap
        )

        easter_day_cap = config["features"]["days_to_easter_cap"]
        easter_date = easter(next_date.year)
        easter_ts = pd.Timestamp(easter_date)
        if next_date > easter_ts:
            easter_ts = pd.Timestamp(easter(next_date.year + 1))
        feature_dict["days_to_easter"] = min(abs((next_date - easter_ts).days), easter_day_cap)

        # Convert to DataFrame row
        feature_row = pd.DataFrame([feature_dict])

        # Predict
        pred_sales = model.predict(feature_row)[0]
        pred_sales = max(0, pred_sales)  # Clip to non-negative

        # Confidence interval
        ci_pct = config["forecaster"]["confidence_interval"]
        lower_bound = pred_sales * (1 - ci_pct)
        upper_bound = pred_sales * (1 + ci_pct)

        forecasts.append({
            "date": next_date,
            "store_id": store_id,
            "forecasted_sales": pred_sales,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "promo": promo_schedule[day_idx] if day_idx < len(promo_schedule) else 0,
        })

        # Add to context for next iteration (auto-regressive)
        feature_dict["sales"] = pred_sales
        new_row = pd.DataFrame([feature_dict])
        context = pd.concat([context, new_row], ignore_index=True)
        context = context.tail(400)  # Keep only last 400 for memory

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


def scenario_simulation(
    model: DemandForecaster,
    df: pd.DataFrame,
    store_id: int,
    horizon_days: int,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Simulate 3 promotion scenarios.

    Args:
        model: Trained DemandForecaster.
        df: Historical DataFrame.
        store_id: Store ID to simulate.
        horizon_days: Number of days to forecast.
        config: Configuration dictionary.

    Returns:
        Dict with scenario names as keys, forecast DataFrames as values:
            {
                "Baseline (No Promo)": DataFrame,
                "Weekly Promo": DataFrame,
                "Aggressive Promo": DataFrame,
            }
    """
    scenarios = {}

    # Scenario 1: No promo
    promo_schedule_no = [0] * horizon_days
    scenarios["Baseline (No Promo)"] = forecast_future(
        model, df, store_id, horizon_days, promo_schedule_no, config
    )

    # Scenario 2: Weekly promo (every 7 days)
    promo_schedule_weekly = [1 if (i % 7 == 0) else 0 for i in range(horizon_days)]
    scenarios["Weekly Promo"] = forecast_future(
        model, df, store_id, horizon_days, promo_schedule_weekly, config
    )

    # Scenario 3: Aggressive promo (every 3 days)
    promo_schedule_aggressive = [1 if (i % 3 == 0) else 0 for i in range(horizon_days)]
    scenarios["Aggressive Promo"] = forecast_future(
        model, df, store_id, horizon_days, promo_schedule_aggressive, config
    )

    return scenarios
