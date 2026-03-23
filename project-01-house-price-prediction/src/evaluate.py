from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils import save_json, ensure_dir

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main(model_path: str, split_path: str, reports_dir: str) -> None:
    pipeline = joblib.load(model_path)
    split = joblib.load(split_path)
    X_test = split["X_test"]
    y_test = split["y_test"]

    preds = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse_val = rmse(y_test, preds)

    metrics = {
        "model_path": model_path,
        "mae": mae,
        "rmse": rmse_val
    }

    reports = Path(reports_dir)
    ensure_dir(reports / "figures")

    # Save metrics.json
    save_json(reports / "metrics.json", metrics)

    # Feature importance / coefficient plot
    model = pipeline.named_steps["model"]
    feature_names = list(X_test.columns)
    fig_path = reports / "figures" / "feature_importance.png"

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 5))
        plt.title("Feature Importance (RandomForest)")
        plt.bar(range(len(feature_names)), importances[order])
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in order], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        print(f"Saved feature importance plot to: {fig_path}")
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        order = np.argsort(coefs)[::-1]

        plt.figure(figsize=(10, 5))
        plt.title("Absolute Coefficients (LinearRegression)")
        plt.bar(range(len(feature_names)), coefs[order])
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in order], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        print(f"Saved coefficient-importance plot to: {fig_path}")

    print("Metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, default="models/test_split.joblib")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()
    main(args.model_path, args.split_path, args.reports_dir)
