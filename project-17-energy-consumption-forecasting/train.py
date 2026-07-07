"""
End-to-end training entry point.

Usage:
    python train.py

Reads config.yaml, loads (or generates) data/energy_consumption.csv,
engineers features, trains every candidate model, picks the best by
test MAE, and writes all reports/figures + the saved model.
"""

from pathlib import Path

import pandas as pd
import yaml

from src.data_generator import generate_energy_data
from src.features import create_features
from src.trainer import train_all_models, select_and_save_best
from src.evaluator import (
    weekday_error_chart,
    forecast_vs_actual_chart,
    model_comparison_chart,
    write_report,
)


def load_or_generate_data(config: dict) -> pd.DataFrame:
    data_path = Path(config["paths"]["data"])
    if data_path.exists():
        print(f"Loading existing dataset -> {data_path}")
        df = pd.read_csv(data_path, parse_dates=["date"])
    else:
        print("No dataset found, generating synthetic data...")
        df = generate_energy_data(config)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    return df


def main():
    config = yaml.safe_load(open("config.yaml"))

    df = load_or_generate_data(config)
    df_feat = create_features(df, config)

    print(f"Training on {len(df_feat)} rows with {len(df_feat.columns) - 2} features...")
    results, feature_cols = train_all_models(df_feat, config)

    for name, r in results.items():
        print(f"  {name:20s} MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  MAPE={r['mape']:.2f}%")

    best_name, metadata = select_and_save_best(results, feature_cols, config)
    print(f"\nBest model: {best_name} -> saved to {config['paths']['best_model']}")

    figures_dir = config["paths"]["figures"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    weekday_mae = weekday_error_chart(results[best_name], figures_dir)
    forecast_vs_actual_chart(results[best_name], figures_dir)
    model_comparison_chart(results, figures_dir)

    report_path = Path(config["paths"]["reports"]) / "evaluation_report.txt"
    write_report(best_name, metadata, weekday_mae, str(report_path))
    print(f"Report written -> {report_path}")


if __name__ == "__main__":
    main()
