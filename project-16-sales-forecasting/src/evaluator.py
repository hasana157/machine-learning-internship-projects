"""Evaluation utilities: metrics + plots, shared by train.py and the app."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_%": round(mape, 3)}


def plot_forecast(test_df: pd.DataFrame, y_test, preds, model_type: str, figures_dir: str) -> str:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(figures_dir) / f"{model_type}_forecast.png"

    plt.figure(figsize=(11, 4.5))
    plt.plot(test_df["date"], y_test.values, label="Actual", linewidth=1.8)
    plt.plot(test_df["date"], preds, label="Forecast", linewidth=1.8, linestyle="--")
    plt.title(f"{model_type.upper()} — Actual vs Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def plot_residuals(test_df: pd.DataFrame, y_test, preds, model_type: str, figures_dir: str) -> str:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(figures_dir) / f"{model_type}_residuals.png"
    residuals = y_test.values - preds

    plt.figure(figsize=(11, 3.5))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(test_df["date"], residuals, marker="o", markersize=3, linewidth=1)
    plt.title(f"{model_type.upper()} — Residuals (Actual - Forecast)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def plot_feature_importance(model, feature_cols, model_type: str, figures_dir: str) -> str | None:
    if not hasattr(model, "feature_importances_"):
        return None

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(figures_dir) / f"{model_type}_feature_importance.png"

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

    plt.figure(figsize=(8, max(4, 0.3 * len(feature_cols))))
    importances.plot(kind="barh")
    plt.title(f"{model_type.upper()} — Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def write_report(metrics_by_model: dict, reports_dir: str) -> str:
    out_path = Path(reports_dir) / "evaluation_report.txt"
    lines = ["Project 16 — Sales Forecasting — Evaluation Report", "=" * 52, ""]
    for model_type, metrics in metrics_by_model.items():
        lines.append(f"[{model_type.upper()}]")
        for k, v in metrics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    logger.info(f"Wrote evaluation report to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    # Standalone entrypoint: re-evaluate whatever models are already saved,
    # assuming `python train.py --model_type both` has been run first.
    import joblib
    from src.utils import load_config
    from src.trainer import load_or_generate_data, chronological_split
    from src.features import create_features, get_feature_columns

    config = load_config()
    df = load_or_generate_data(config)
    df_feat = create_features(df, config["features"]["lags"], config["features"]["windows"])
    feature_cols = get_feature_columns(df_feat)
    train_df, test_df = chronological_split(df_feat, config["evaluation"]["train_split"])
    y_test = df_feat.loc[test_df.index, "sales"]

    metrics_by_model = {}
    for model_type in ("linear", "rf"):
        model_path = Path(config["paths"]["models_dir"]) / f"{model_type}_model.joblib"
        if not model_path.exists():
            logger.warning(f"Skipping {model_type}: no saved model at {model_path}")
            continue
        model = joblib.load(model_path)
        preds = model.predict(df_feat.loc[test_df.index, feature_cols])
        metrics_by_model[model_type] = compute_metrics(y_test, preds)
        plot_forecast(test_df, y_test, preds, model_type, config["paths"]["figures_dir"])
        plot_residuals(test_df, y_test, preds, model_type, config["paths"]["figures_dir"])
        plot_feature_importance(model, feature_cols, model_type, config["paths"]["figures_dir"])

    write_report(metrics_by_model, config["paths"]["reports_dir"])
    print(metrics_by_model)
