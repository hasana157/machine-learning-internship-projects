"""
CLI training entry point.

Usage:
    python train.py --config config.yaml

Runs the full pipeline: ingest -> validate -> engineer features ->
split -> tune 3 candidate models -> select best by CV ROC-AUC ->
evaluate on held-out test set -> save pipeline, metrics, figures,
and (optionally) log everything to MLflow.
"""

import argparse
from pathlib import Path

from src.data_ingestion import load_data
from src.evaluator import (
    compute_metrics, plot_calibration, plot_feature_importance,
    plot_precision_recall, plot_roc_curves, write_metrics_report,
)
from src.explainer import save_background_sample, save_global_summary
from src.features import engineer_features
from src.preprocessing import get_feature_names
from src.trainer import (
    save_pipeline, save_training_summary, select_best, split_data, tune_candidates,
)
from src.utils import get_logger, load_config

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the loan approval risk model.")
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Path to the YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger.info("=== Step 1/6: Data ingestion & validation ===")
    df = load_data(config)

    logger.info("=== Step 2/6: Feature engineering ===")
    df = engineer_features(df)

    logger.info("=== Step 3/6: Train/test split ===")
    X_train, X_test, y_train, y_test = split_data(df, config)
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
                f"Positive rate (train): {y_train.mean():.2%}")

    logger.info("=== Step 4/6: Hyperparameter tuning (per candidate model) ===")
    results = tune_candidates(X_train, y_train, config)
    best_name, best_result = select_best(results)
    best_pipeline = best_result["best_estimator"]
    logger.info(f"Best model: {best_name} (CV {config['training']['scoring']}"
                f"={best_result['best_cv_score']:.4f})")

    logger.info("=== Step 5/6: Held-out evaluation ===")
    metrics_by_model = {}
    fitted_models = {}
    for name, r in results.items():
        model = r["best_estimator"]
        fitted_models[name] = model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics_by_model[name] = compute_metrics(y_test, y_pred, y_proba)
        logger.info(f"  {name}: ROC-AUC={metrics_by_model[name]['roc_auc']:.4f} "
                    f"PR-AUC={metrics_by_model[name]['pr_auc']:.4f} "
                    f"F1={metrics_by_model[name]['f1_score']:.4f}")

    figures_dir = config["paths"]["figures_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    plot_roc_curves(fitted_models, X_test, y_test, figures_dir)
    plot_precision_recall(best_pipeline, X_test, y_test, figures_dir)
    plot_calibration(best_pipeline, X_test, y_test, figures_dir)

    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    plot_feature_importance(best_pipeline, best_name, feature_names, figures_dir, artifacts_dir)

    logger.info("=== Step 6/6: Explainability (SHAP) + saving artifacts ===")
    save_global_summary(best_pipeline, X_test, feature_names, figures_dir)
    save_background_sample(best_pipeline, X_train, f"{artifacts_dir}/shap_background.joblib")

    write_metrics_report(metrics_by_model, best_name, config["paths"]["metrics"])
    save_pipeline(best_pipeline, config)
    save_training_summary(results, best_name, config)

    logger.info("Training complete. Run `python evaluate.py` any time to regenerate "
                "reports from the saved pipeline, or `python predict.py` for a sample prediction.")


if __name__ == "__main__":
    main()
