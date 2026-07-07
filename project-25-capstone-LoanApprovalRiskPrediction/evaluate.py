"""
Standalone evaluation CLI.

Reloads the saved production pipeline, rebuilds the identical
held-out test split (same random_state as training), and prints a
business-readable evaluation report. Useful for re-checking model
health without rerunning the full (slower) training + tuning step.

Usage:
    python evaluate.py --config config.yaml
"""

import argparse
import json

import joblib

from src.data_ingestion import load_data
from src.evaluator import compute_metrics
from src.features import engineer_features
from src.trainer import split_data
from src.utils import get_logger, load_config

logger = get_logger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the saved loan risk pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def print_report(metrics: dict, best_name: str):
    print("\n" + "=" * 60)
    print("LOAN APPROVAL RISK MODEL — EVALUATION REPORT")
    print("=" * 60)
    print(f"Model: {best_name}\n")
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}   (ranking quality of risk scores)")
    print(f"  PR-AUC        : {metrics['pr_auc']:.4f}   (precision/recall trade-off on minority class)")
    print(f"  F1-score      : {metrics['f1_score']:.4f}   (balance of precision & recall at 0.5 threshold)")
    print(f"  Brier score   : {metrics['brier_score']:.4f}   (lower = better-calibrated probabilities)")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion matrix [[TN, FP], [FN, TP]]: {cm}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    config = load_config(args.config)

    logger.info("Loading saved pipeline...")
    pipeline = joblib.load(config["paths"]["best_pipeline"])

    logger.info("Rebuilding identical held-out test split...")
    df = load_data(config)
    df = engineer_features(df)
    _, X_test, _, y_test = split_data(df, config)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    with open(config["paths"]["metrics"]) as f:
        best_name = json.load(f)["best_model"]

    print_report(metrics, best_name)


if __name__ == "__main__":
    main()
