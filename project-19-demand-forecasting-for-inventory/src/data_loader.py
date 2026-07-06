"""
Data loading and cleaning for Rossmann Store Sales dataset.

Supports two modes:
    - MODE A: Load real Kaggle CSV files (train.csv + store.csv)
    - MODE B: Generate synthetic Rossmann-like data for demo/fallback

Auto-detects which mode to use based on file existence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date

logger = logging.getLogger(__name__)


def load_rossmann(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and clean real Rossmann Kaggle data.

    Process:
        1. Load train.csv and store.csv
        2. Parse Date as datetime
        3. Filter: keep only Open == 1 rows
        4. Filter: keep only Sales > 0 rows
        5. Merge train and store on Store column
        6. Rename columns to snake_case
        7. Fill missing competition/promo metadata
        8. Convert state_holiday to numeric
        9. Sort by store_id, date
        10. Return cleaned DataFrame

    Args:
        config: Configuration dictionary containing data paths.

    Returns:
        Cleaned pandas DataFrame with columns:
            [store_id, date, sales, customers, open, promo, state_holiday,
             school_holiday, store_type, assortment, competition_distance,
             competition_open_since_month, competition_open_since_year,
             promo2, promo2_since_week, promo2_since_year, promo_interval]

    Raises:
        FileNotFoundError: If train.csv or store.csv not found.
    """
    train_path = Path(config["data"]["kaggle_train"])
    store_path = Path(config["data"]["kaggle_store"])

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not store_path.exists():
        raise FileNotFoundError(f"Store file not found: {store_path}")

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path, dtype={"StateHoliday": str}, low_memory=False)

    logger.info(f"Loading store metadata from {store_path}")
    store_df = pd.read_csv(store_path)

    # Parse date
    train_df["Date"] = pd.to_datetime(train_df["Date"])

    # Filter: keep only open stores
    initial_rows = len(train_df)
    train_df = train_df[train_df["Open"] == 1].copy()
    logger.info(f"Filtered out closed stores: {initial_rows:,} → {len(train_df):,} rows")

    # Filter: keep only positive sales
    initial_rows = len(train_df)
    train_df = train_df[train_df["Sales"] > 0].copy()
    logger.info(f"Filtered out zero sales: {initial_rows:,} → {len(train_df):,} rows")

    # Merge train and store metadata
    df = train_df.merge(store_df, on="Store", how="left")

    # Rename columns to snake_case
    column_mapping = {
        "Store": "store_id",
        "Sales": "sales",
        "Date": "date",
        "Promo": "promo",
        "Customers": "customers",
        "Open": "open",
        "StateHoliday": "state_holiday",
        "SchoolHoliday": "school_holiday",
        "StoreType": "store_type",
        "Assortment": "assortment",
        "CompetitionDistance": "competition_distance",
        "CompetitionOpenSinceMonth": "competition_open_since_month",
        "CompetitionOpenSinceYear": "competition_open_since_year",
        "Promo2": "promo2",
        "Promo2SinceWeek": "promo2_since_week",
        "Promo2SinceYear": "promo2_since_year",
        "PromoInterval": "promo_interval",
    }
    df.rename(columns=column_mapping, inplace=True)

    # Fill missing competition distance with median
    if "competition_distance" in df.columns:
        median_comp_dist = df["competition_distance"].median()
        df["competition_distance"] = df["competition_distance"].fillna(median_comp_dist)
        logger.info(f"Filled missing competition_distance with median: {median_comp_dist:.0f}")

    # Fill promo2 metadata with 0
    promo2_cols = ["promo2_since_week", "promo2_since_year"]
    for col in promo2_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Convert state_holiday to numeric
    state_holiday_map = {"0": 0, "a": 1, "b": 2, "c": 3}
    if "state_holiday" in df.columns:
        df["state_holiday"] = df["state_holiday"].astype(str).map(state_holiday_map)
        df["state_holiday"] = df["state_holiday"].fillna(0)

    # Sort by store_id and date
    df.sort_values(by=["store_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Log summary
    logger.info(f"✅ Rossmann data loaded: {len(df):,} rows")
    logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"🏪 Unique stores: {df['store_id'].nunique()}")

    return df


def generate_synthetic_rossmann(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate synthetic Rossmann-like data for demo/fallback mode.

    Creates realistic sales patterns including:
        - Weekly seasonality (weekend boost)
        - Trend component
        - Promotional uplift
        - Holiday dips
        - Gaussian noise

    Args:
        config: Configuration dictionary.

    Returns:
        Synthetic DataFrame with identical column structure to load_rossmann().
    """
    n_stores = config["data"]["synthetic_stores"]
    start_date_str = config["data"]["start_date"]
    end_date_str = config["data"]["end_date"]

    start_date = parse_date(start_date_str).date()
    end_date = parse_date(end_date_str).date()

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)

    logger.warning("⚠️  Kaggle data not found — using synthetic fallback.")
    logger.warning(
        "📥 Download from: kaggle.com/competitions/rossmann-store-sales for full results."
    )

    rng = np.random.RandomState(42)

    records = []

    store_types = ["a", "b", "c", "d"]
    assortments = ["a", "b", "c"]

    for store_id in range(1, n_stores + 1):
        # Base sales for this store
        base_sales = rng.uniform(3000, 15000)

        # Trend over time
        trend = np.linspace(0, rng.uniform(-500, 2000), n_days)

        for day_idx, date in enumerate(date_range):
            dow = date.dayofweek  # 0=Monday, 6=Sunday
            day_of_month = date.day
            month = date.month
            year = date.year
            week_of_year = date.isocalendar()[1]
            quarter = (month - 1) // 3 + 1

            # Weekly seasonality (weekend boost)
            if dow >= 5:  # Saturday or Sunday
                weekly_component = base_sales * 0.3 * 0.8  # Slight boost
            else:
                weekly_component = base_sales * 0.3 * np.sin(2 * np.pi * dow / 7)

            # Promo effect
            promo = rng.choice([0, 1], p=[0.85, 0.15])
            promo_boost = (rng.uniform(0.10, 0.30) * base_sales) if promo == 1 else 0

            # State holiday effect (roughly 4 days per year)
            state_holiday = 0
            if (month == 12 and day_of_month >= 24) or (month == 1 and day_of_month <= 2):
                state_holiday = 3  # Christmas
                holiday_dip = -0.4 * base_sales
            elif month == 4 and 9 <= day_of_month <= 11:  # Approximate Easter
                state_holiday = 2
                holiday_dip = -0.3 * base_sales
            else:
                holiday_dip = 0
                if rng.random() < 0.02:  # Random public holiday
                    state_holiday = 1
                    holiday_dip = -0.2 * base_sales

            school_holiday = 1 if month in [7, 8] else 0  # Summer break

            # Combine components
            noise = rng.normal(0, base_sales * 0.05)
            sales = base_sales + trend[day_idx] + weekly_component + promo_boost + holiday_dip + noise
            sales = max(500, sales)  # Clip to minimum

            customers = int(sales / rng.uniform(15, 25))
            customers = max(10, customers)

            records.append({
                "store_id": store_id,
                "date": date,
                "sales": sales,
                "customers": customers,
                "open": 1,
                "promo": promo,
                "state_holiday": state_holiday,
                "school_holiday": school_holiday,
                "store_type": rng.choice(store_types),
                "assortment": rng.choice(assortments),
                "competition_distance": rng.uniform(100, 30000),
                "competition_open_since_month": rng.randint(1, 13),
                "competition_open_since_year": rng.randint(2000, 2020),
                "promo2": rng.choice([0, 1], p=[0.6, 0.4]),
                "promo2_since_week": rng.randint(1, 53),
                "promo2_since_year": rng.randint(2005, 2020),
                "promo_interval": rng.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec", None]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["store_id", "date"]).reset_index(drop=True)

    logger.info(f"✅ Synthetic data generated: {len(df):,} rows")
    logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"🏪 Unique stores: {df['store_id'].nunique()}")

    return df


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    Auto-routing data loader: use Kaggle data if available, else synthetic.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (DataFrame, data_source_str) where data_source_str is
        either "kaggle" or "synthetic".
    """
    train_path = Path(config["data"]["kaggle_train"])

    if train_path.exists():
        logger.info("📦 Using Kaggle data")
        df = load_rossmann(config)
        return df, "kaggle"
    else:
        logger.info("📦 Using synthetic fallback data")
        df = generate_synthetic_rossmann(config)
        return df, "synthetic"
