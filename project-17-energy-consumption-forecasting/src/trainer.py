"""
Training pipeline: chronological train/test split, multi-model
comparison, and best-model persistence.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import create_features, feature_columns
from src.models import build_candidates


def chronological_split(df: pd.DataFrame, train_split: float):
    split_idx = int(len(df) * train_split)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_all_models(df_feat: pd.DataFrame, config: dict):
    """Train every candidate model and return per-model results."""
    cols = feature_columns(df_feat)
    train_df, test_df = chronological_split(df_feat, config["model"]["train_split"])

    X_train, y_train = train_df[cols], train_df["consumption"]
    X_test, y_test = test_df[cols], test_df["consumption"]

    candidates = build_candidates(config)
    results = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = float(np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100)

        results[name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "preds": preds,
            "y_test": y_test.values,
            "dates_test": test_df["date"].values,
        }

    return results, cols


def select_and_save_best(results: dict, feature_cols: list, config: dict):
    best_name = min(results, key=lambda k: results[k]["mae"])
    best = results[best_name]

    paths = config["paths"]
    Path(paths["models_dir"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], paths["best_model"])

    metadata = {
        "best_model": best_name,
        "feature_columns": feature_cols,
        "metrics": {
            name: {"mae": r["mae"], "rmse": r["rmse"], "mape": r["mape"]}
            for name, r in results.items()
        },
    }
    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    return best_name, metadata
