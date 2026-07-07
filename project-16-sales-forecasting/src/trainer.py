"""Training orchestration for Project 16 — Sales Forecasting.

Handles: loading (or generating) data, feature engineering,
a chronological train/test split, fitting the chosen model, and
persisting the fitted model + the exact feature column order.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from src.data_generator import generate_sales_data
from src.features import create_features, get_feature_columns
from src.model import build_model
from src.utils import get_logger

logger = get_logger(__name__)


def load_or_generate_data(config: dict) -> pd.DataFrame:
    """Load data/sales_data.csv if present, otherwise generate it.

    This lets users drop in a real CSV (see README) without touching
    any code — the file is only synthesized when it's missing.
    """
    data_path = Path(config["paths"]["data"])
    if data_path.exists():
        logger.info(f"Loading existing dataset from {data_path}")
        return pd.read_csv(data_path, parse_dates=["date"])

    logger.info("No dataset found — generating synthetic sales data")
    df = generate_sales_data(config)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    return df


def chronological_split(df: pd.DataFrame, train_split: float):
    """Split by position, never by shuffling — respects temporal order."""
    split_idx = int(len(df) * train_split)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_model(model_type: str, config: dict):
    """Full pipeline for one model type. Returns a dict of run artifacts."""
    df = load_or_generate_data(config)
    df_feat = create_features(
        df,
        lags=config["features"]["lags"],
        windows=config["features"]["windows"],
    )

    feature_cols = get_feature_columns(df_feat)
    X = df_feat[feature_cols]
    y = df_feat["sales"]

    train_df, test_df = chronological_split(df_feat, config["evaluation"]["train_split"])
    X_train, X_test = X.loc[train_df.index], X.loc[test_df.index]
    y_train, y_test = y.loc[train_df.index], y.loc[test_df.index]

    model = build_model(model_type, config)
    logger.info(f"Fitting {model_type} on {len(X_train)} rows / {len(feature_cols)} features")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_type}_model.joblib"
    joblib.dump(model, model_path)

    feature_names_path = models_dir / f"{model_type}_feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    logger.info(f"Saved model to {model_path}")

    return {
        "model": model,
        "model_type": model_type,
        "feature_cols": feature_cols,
        "df_feat": df_feat,
        "test_df": test_df,
        "y_test": y_test,
        "preds": preds,
        "model_path": str(model_path),
    }
