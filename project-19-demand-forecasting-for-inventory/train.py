#!/usr/bin/env python3
"""
ForecastIQ Training Script

CLI entry point for training the demand forecasting model.
Orchestrates full pipeline: load data → engineer features → train → evaluate → save.

Usage:
    python train.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
from src.data_loader import load_data
from src.features import engineer_features, get_numeric_feature_cols, get_categorical_feature_cols, get_target_col
from src.model import DemandForecaster
from src.evaluator import compute_metrics, compute_per_store_metrics, generate_evaluation_figures, generate_evaluation_report
from src.utils import load_config, ensure_directory, setup_logger

# Setup logger
logger = setup_logger(__name__)


def main() -> None:
    """Run full training pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 ForecastIQ Training Pipeline Started")
    logger.info("=" * 60)

    # Load configuration
    config = load_config("config.yaml")
    logger.info("📋 Configuration loaded")

    # Ensure output directories
    ensure_directory(config["paths"]["reports"])
    ensure_directory(config["paths"]["figures"])

    # Load data
    df, data_source = load_data(config)
    logger.info(f"📦 Dataset: {data_source}")
    logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"🏪 Stores: {df['store_id'].nunique()}")
    logger.info(f"📊 Total rows: {len(df):,}")

    # Feature engineering
    df_feat = engineer_features(df, config)

    # Train/test split (time-based, no shuffling)
    split_ratio = config["evaluation"]["train_split_ratio"]
    split_date = df_feat["date"].quantile(split_ratio)
    train_df = df_feat[df_feat["date"] <= split_date].copy()
    test_df = df_feat[df_feat["date"] > split_date].copy()

    logger.info(f"✂️  Train/test split: {len(train_df):,} / {len(test_df):,} rows")

    # Prepare X, y
    X_train = train_df.drop(columns=["sales", "date"])
    y_train = train_df["sales"]
    X_test = test_df.drop(columns=["sales", "date"])
    y_test = test_df["sales"]

    # Train model
    logger.info("=" * 60)
    logger.info("🏋️  Training Models")
    logger.info("=" * 60)

    model = DemandForecaster(config["model"])
    model.fit(X_train, y_train, data_source=data_source)

    # Predictions
    preds_rf = model.predict(X_test)
    preds_baseline = model.predict_baseline(X_test)

    # Metrics
    metrics = compute_metrics(y_test.values, preds_rf, preds_baseline)

    logger.info("=" * 60)
    logger.info("📊 Test Set Metrics")
    logger.info("=" * 60)
    logger.info(f"🔴 Random Forest:")
    logger.info(f"   MAE:    {metrics['rf_mae']:.2f}")
    logger.info(f"   RMSE:   {metrics['rf_rmse']:.2f}")
    logger.info(f"   MAPE:   {metrics['rf_mape']:.2f}%")
    logger.info(f"   RMSPE:  {metrics['rf_rmspe']:.4f}  ← Kaggle metric")
    logger.info(f"   R²:     {metrics['rf_r2']:.4f}")

    logger.info(f"🔵 Linear Baseline:")
    logger.info(f"   MAE:    {metrics['baseline_mae']:.2f}")
    logger.info(f"   RMSE:   {metrics['baseline_rmse']:.2f}")
    logger.info(f"   MAPE:   {metrics['baseline_mape']:.2f}%")
    logger.info(f"   RMSPE:  {metrics['baseline_rmspe']:.4f}")
    logger.info(f"   R²:     {metrics['baseline_r2']:.4f}")

    improvement = (metrics['baseline_rmspe'] - metrics['rf_rmspe']) / metrics['baseline_rmspe'] * 100
    logger.info(f"✨ RF improvement over baseline RMSPE: {improvement:+.1f}%")

    # Save model
    logger.info("=" * 60)
    logger.info("💾 Saving Artifacts")
    logger.info("=" * 60)

    model_path = config["paths"]["model"]
    model.save(model_path)

    # Save test predictions
    import pandas as pd

    test_preds_df = pd.DataFrame({
        "date": test_df["date"].values,
        "store_id": test_df["store_id"].values,
        "actual_sales": y_test.values,
        "predicted_sales": preds_rf,
        "abs_error": abs(y_test.values - preds_rf),
        "pct_error": (abs(y_test.values - preds_rf) / y_test.values * 100),
    })
    test_preds_path = f"{config['paths']['reports']}/test_predictions.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    logger.info(f"✅ Test predictions saved to {test_preds_path}")

    # Save per-store metrics
    store_metrics_df = compute_per_store_metrics(test_df, y_test.values, preds_rf)
    store_metrics_path = f"{config['paths']['reports']}/store_metrics.csv"
    store_metrics_df.to_csv(store_metrics_path, index=False)
    logger.info(f"✅ Store metrics saved to {store_metrics_path}")

    # Generate evaluation figures
    logger.info("=" * 60)
    logger.info("📊 Generating Evaluation Figures")
    logger.info("=" * 60)

    generate_evaluation_figures(
        test_df, y_test.values, preds_rf, preds_baseline, model, metrics, config["paths"]["figures"]
    )

    # Generate report
    generate_evaluation_report(test_df, y_test.values, preds_rf, metrics, data_source, config["paths"]["reports"])

    logger.info("=" * 60)
    logger.info("✅ ForecastIQ Training Complete!")
    logger.info("=" * 60)
    logger.info(f"📈 RMSPE: {metrics['rf_rmspe']:.4f}")
    logger.info(f"📦 Model saved to: {model_path}")
    logger.info(f"📊 Reports saved to: {config['paths']['reports']}")
    logger.info("🚀 Next: python forecast.py --store 1 --days 30")
    logger.info("💻 Dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
