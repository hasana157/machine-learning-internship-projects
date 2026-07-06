"""
Training pipeline for DemandForecaster.

Orchestrates:
    1. Data loading and cleaning
    2. Feature engineering
    3. Train/test split (time-series aware)
    4. Model training
    5. Evaluation and metric computation
    6. Model persistence
"""

import logging
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_data
from evaluator import compute_metrics
from features import engineer_features, get_numeric_feature_cols, get_categorical_feature_cols, get_target_col
from model import DemandForecaster
from utils import load_config, ensure_directory

logger = logging.getLogger(__name__)


def train_demand_forecaster(config_path: str = "config.yaml") -> Tuple[DemandForecaster, Dict[str, Any]]:
    """
    Full training pipeline: load data → engineer features → train model → evaluate.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Tuple of (trained_model, evaluation_dict) where evaluation_dict contains metrics.
    """
    # Load configuration
    config = load_config(config_path)
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
    split_date = df_feat["date"].quantile(config["evaluation"]["train_split_ratio"])
    train_df = df_feat[df_feat["date"] <= split_date].copy()
    test_df = df_feat[df_feat["date"] > split_date].copy()

    logger.info(f"✂️  Train/test split: {len(train_df):,} / {len(test_df):,} rows")

    # Prepare X, y
    X_train = train_df.drop(columns=["sales", "date"])
    y_train = train_df["sales"]
    X_test = test_df.drop(columns=["sales", "date"])
    y_test = test_df["sales"]

    # Train model
    model = DemandForecaster(config["model"])
    model.fit(X_train, y_train, data_source=data_source)

    # Predictions
    preds_rf = model.predict(X_test)
    preds_baseline = model.predict_baseline(X_test)

    # Metrics
    metrics = compute_metrics(y_test.values, preds_rf, preds_baseline)

    logger.info(f"📊 RF MAE: {metrics['rf_mae']:.2f}")
    logger.info(f"📊 RF RMSE: {metrics['rf_rmse']:.2f}")
    logger.info(f"📊 RF MAPE: {metrics['rf_mape']:.2f}%")
    logger.info(f"📊 RF RMSPE: {metrics['rf_rmspe']:.4f}")
    logger.info(f"📊 RF R²: {metrics['rf_r2']:.4f}")

    # Save model
    model_path = config["paths"]["model"]
    model.save(model_path)

    # Save test predictions
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
    logger.info(f"💾 Test predictions saved to {test_preds_path}")

    logger.info(f"✅ ForecastIQ model trained. RMSPE: {metrics['rf_rmspe']:.4f}")

    return model, metrics
