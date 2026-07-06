#!/usr/bin/env python3
"""
Step-by-step training script with detailed progress monitoring.
Allows monitoring each stage of the training pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
import time
from src.data_loader import load_data
from src.features import engineer_features, get_numeric_feature_cols, get_categorical_feature_cols, get_target_col
from src.model import DemandForecaster
from src.evaluator import compute_metrics, compute_per_store_metrics, generate_evaluation_figures, generate_evaluation_report
from src.utils import load_config, ensure_directory, setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Run training pipeline step-by-step with timing."""
    
    logger.info("=" * 70)
    logger.info("🚀 ForecastIQ Training Pipeline (Step-by-Step)")
    logger.info("=" * 70)

    # STEP 1: Load configuration
    logger.info("\n📋 STEP 1: Loading configuration...")
    start_time = time.time()
    config = load_config("config.yaml")
    logger.info(f"✓ Configuration loaded in {time.time() - start_time:.2f}s")
    
    # Ensure output directories
    ensure_directory(config["paths"]["reports"])
    ensure_directory(config["paths"]["figures"])

    # STEP 2: Load data
    logger.info("\n📦 STEP 2: Loading data...")
    start_time = time.time()
    df, data_source = load_data(config)
    elapsed = time.time() - start_time
    logger.info(f"✓ Data loaded in {elapsed:.2f}s")
    logger.info(f"  - Dataset: {data_source}")
    logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  - Stores: {df['store_id'].nunique()}")
    logger.info(f"  - Total rows: {len(df):,}")
    logger.info(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # STEP 3: Feature Engineering
    logger.info("\n🔧 STEP 3: Engineering features...")
    start_time = time.time()
    df_features = engineer_features(df, config)
    elapsed = time.time() - start_time
    logger.info(f"✓ Features engineered in {elapsed:.2f}s")
    logger.info(f"  - Total features: {len(df_features.columns)}")
    logger.info(f"  - Shape: {df_features.shape}")
    
    # Get feature column names
    numeric_cols = get_numeric_feature_cols()
    categorical_cols = get_categorical_feature_cols()
    target_col = get_target_col()
    logger.info(f"  - Numeric features: {len(numeric_cols)}")
    logger.info(f"  - Categorical features: {len(categorical_cols)}")

    # STEP 4: Train/Test Split
    logger.info("\n✂️  STEP 4: Train/test split...")
    start_time = time.time()
    split_idx = int(len(df_features) * config["evaluation"]["train_split_ratio"])
    X_train = df_features.iloc[:split_idx, :]
    X_test = df_features.iloc[split_idx:, :]
    
    y_train = X_train[target_col].copy()
    y_test = X_test[target_col].copy()
    
    X_train = X_train.drop(columns=[target_col])
    X_test = X_test.drop(columns=[target_col])
    elapsed = time.time() - start_time
    
    logger.info(f"✓ Train/test split done in {elapsed:.2f}s")
    logger.info(f"  - Train shape: {X_train.shape}")
    logger.info(f"  - Test shape: {X_test.shape}")
    logger.info(f"  - Train/test ratio: {len(X_train)}/{len(X_test)}")

    # STEP 5: Initialize Model
    logger.info("\n🤖 STEP 5: Initializing model...")
    start_time = time.time()
    model = DemandForecaster(config["model"])
    elapsed = time.time() - start_time
    logger.info(f"✓ Model initialized in {elapsed:.2f}s")
    logger.info(f"  - Estimators: {config['model']['n_estimators']}")
    logger.info(f"  - Max depth: {config['model']['max_depth']}")

    # STEP 6: Train Model
    logger.info("\n🏋️  STEP 6: Training model (this may take a while)...")
    start_time = time.time()
    try:
        model.fit(X_train, y_train, data_source=data_source)
        elapsed = time.time() - start_time
        logger.info(f"✓ Model trained in {elapsed/60:.2f} minutes")
    except KeyboardInterrupt:
        logger.error("⚠️  Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

    # STEP 7: Evaluate Model
    logger.info("\n📊 STEP 7: Evaluating model...")
    start_time = time.time()
    
    # Train metrics
    train_metrics = compute_metrics(
        y_true=y_train,
        y_pred=model.predict(X_train),
        set_name="Train"
    )
    logger.info(f"  - Train RMSE: {train_metrics['rmse']:.4f}")
    logger.info(f"  - Train MAE: {train_metrics['mae']:.4f}")
    logger.info(f"  - Train MAPE: {train_metrics['mape']:.4f}%")
    
    # Test metrics
    test_metrics = compute_metrics(
        y_true=y_test,
        y_pred=model.predict(X_test),
        set_name="Test"
    )
    logger.info(f"  - Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  - Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  - Test MAPE: {test_metrics['mape']:.4f}%")
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Evaluation done in {elapsed:.2f}s")

    # STEP 8: Per-Store Metrics
    logger.info("\n🏪 STEP 8: Computing per-store metrics...")
    start_time = time.time()
    per_store = compute_per_store_metrics(
        y_true=y_test,
        y_pred=model.predict(X_test),
        store_ids=X_test['store_id']
    )
    elapsed = time.time() - start_time
    logger.info(f"✓ Per-store metrics computed in {elapsed:.2f}s")
    logger.info(f"  - Stores analyzed: {len(per_store)}")
    logger.info(f"  - Best MAPE: {per_store['mape'].min():.4f}%")
    logger.info(f"  - Worst MAPE: {per_store['mape'].max():.4f}%")

    # STEP 9: Generate Figures
    logger.info("\n📈 STEP 9: Generating evaluation figures...")
    start_time = time.time()
    generate_evaluation_figures(
        y_test=y_test,
        y_pred=model.predict(X_test),
        per_store_metrics=per_store,
        output_dir=config["paths"]["figures"]
    )
    elapsed = time.time() - start_time
    logger.info(f"✓ Figures generated in {elapsed:.2f}s")

    # STEP 10: Save Model
    logger.info("\n💾 STEP 10: Saving model...")
    start_time = time.time()
    model.save(config["paths"]["model"], config["paths"]["metadata"])
    elapsed = time.time() - start_time
    logger.info(f"✓ Model saved in {elapsed:.2f}s")
    logger.info(f"  - Model: {config['paths']['model']}")
    logger.info(f"  - Metadata: {config['paths']['metadata']}")

    # FINAL Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ Training pipeline completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
