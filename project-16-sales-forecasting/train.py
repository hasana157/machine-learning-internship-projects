"""Project 16 — Sales Forecasting — CLI training entrypoint.

Usage:
    python train.py --model_type linear
    python train.py --model_type rf
    python train.py --model_type both      # trains + compares both
"""

import argparse

from src.utils import load_config, ensure_dirs, get_logger
from src.trainer import train_model
from src.evaluator import (
    compute_metrics,
    plot_forecast,
    plot_residuals,
    plot_feature_importance,
    write_report,
)

logger = get_logger(__name__)


def run(model_type: str, config: dict) -> dict:
    result = train_model(model_type, config)
    metrics = compute_metrics(result["y_test"], result["preds"])
    logger.info(f"{model_type} metrics: {metrics}")

    figures_dir = config["paths"]["figures_dir"]
    plot_forecast(result["test_df"], result["y_test"], result["preds"], model_type, figures_dir)
    plot_residuals(result["test_df"], result["y_test"], result["preds"], model_type, figures_dir)
    plot_feature_importance(result["model"], result["feature_cols"], model_type, figures_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a sales forecasting model.")
    parser.add_argument(
        "--model_type",
        choices=["linear", "rf", "both"],
        required=True,
        help="Which model to train: 'linear', 'rf', or 'both'.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)

    model_types = ["linear", "rf"] if args.model_type == "both" else [args.model_type]

    metrics_by_model = {}
    for mt in model_types:
        metrics_by_model[mt] = run(mt, config)

    write_report(metrics_by_model, config["paths"]["reports_dir"])

    print("\n=== Summary ===")
    for mt, m in metrics_by_model.items():
        print(f"{mt:>8}: {m}")


if __name__ == "__main__":
    main()
