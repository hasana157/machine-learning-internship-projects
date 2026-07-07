#!/usr/bin/env python3
"""
MarketSentinel Training CLI

Train stock movement prediction models with walk-forward validation.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.utils import load_config, ensure_directories, print_section
from src.data_loader import (
    download_kaggle_dataset, load_stock_data, get_available_tickers,
    generate_synthetic_stock_data, save_processed_data
)
from src.features import engineer_features
from src.model import StockMovementPredictor
from src.trainer import (
    walk_forward_validation, train_test_split_validation,
    save_model_artifacts
)
from src.evaluator import (
    generate_classification_report, plot_confusion_matrix,
    plot_roc_curve, plot_feature_importance, save_metrics_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(
        description="Train MarketSentinel stock prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --ticker AAPL --mode walk_forward
  python train.py --ticker GOOGL --mode train_test_split --start-date 2021-01-01
  python train.py --ticker TSLA --mode full_retrain
        """
    )
    
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--mode', choices=['walk_forward', 'train_test_split', 'full_retrain'],
                       default='walk_forward', help='Training mode')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    print_section("🚀 MarketSentinel - Model Training Pipeline")
    
    # Load configuration
    logger.info(f"📖 Loading configuration from {args.config}")
    config = load_config(args.config)
    ensure_directories(config)
    
    ticker = args.ticker.upper()
    
    logger.info(f"📊 Training for ticker: {ticker}")
    logger.info(f"   Mode: {args.mode}")
    if args.start_date:
        logger.info(f"   Start: {args.start_date}")
    if args.end_date:
        logger.info(f"   End: {args.end_date}")
    
    # Download/load data
    data_path = config['paths']['data_raw']
    
    # Check if data exists
    available = get_available_tickers(data_path)
    
    if ticker not in available:
        logger.warning(f"⚠️  {ticker} not found in {data_path}")
        logger.info(f"📥 Attempting to download from Kaggle...")
        
        if not download_kaggle_dataset(config['kaggle']['dataset_name'], data_path):
            logger.warning(f"⚠️  Kaggle download failed. Using synthetic data.")
            df = generate_synthetic_stock_data(
                ticker,
                n_days=config['data']['synthetic_config']['n_days'],
                config=config['data']['synthetic_config']
            )
        else:
            df = load_stock_data(ticker, data_path, args.start_date, args.end_date)
    else:
        logger.info(f"✅ Loading data for {ticker}")
        df = load_stock_data(ticker, data_path, args.start_date, args.end_date)
    
    if df.empty:
        logger.error(f"❌ No data available for {ticker}")
        sys.exit(1)
    
    # Feature engineering
    logger.info(f"🔧 Engineering features...")
    df_features = engineer_features(df, config)
    
    if df_features.empty:
        logger.error(f"❌ No features generated")
        sys.exit(1)
    
    # Save processed data
    save_processed_data(df_features, ticker, config['paths']['data_processed'])
    
    # Initialize model
    model = StockMovementPredictor(config)
    
    # Training based on mode
    start_time = datetime.now()
    
    if args.mode == 'walk_forward':
        logger.info(f"\n🔄 Starting walk-forward validation...")
        predictions_df = walk_forward_validation(df_features, model, config)
        
        # Save predictions
        pred_path = Path(config['paths']['data_processed']) / f"{ticker}_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"✅ Saved predictions to {pred_path}")
        
        # Generate metrics
        y_true = predictions_df['actual'].values
        y_pred_rf = predictions_df['prediction_rf'].values
        y_proba_rf = predictions_df['probability_up_rf'].values
        y_pred_lr = predictions_df['prediction_lr'].values
        y_proba_lr = predictions_df['probability_up_lr'].values
        
        metrics = generate_classification_report(
            y_true, y_pred_rf, y_proba_rf, y_pred_lr, y_proba_lr, ticker
        )
        
    elif args.mode == 'train_test_split':
        logger.info(f"\n📚 Starting train-test split validation...")
        metrics = train_test_split_validation(df_features, model, config)
        
        y_true = df_features.iloc[int(len(df_features) * 0.8):]['target'].values
        y_pred_rf = model.predict(df_features.iloc[int(len(df_features) * 0.8):].drop(columns=['target']))
        y_proba_rf = model.predict_proba(df_features.iloc[int(len(df_features) * 0.8):].drop(columns=['target']))[:, 1]
        
    elif args.mode == 'full_retrain':
        logger.info(f"\n🏋️  Starting full retrain on all data...")
        X = df_features.drop(columns=['target'])
        y = df_features['target']
        model.fit(X, y, ticker=ticker)
        logger.info(f"✅ Trained on all {len(X)} samples")
        metrics = {'full_data': {'samples': len(X)}}
    
    # Save model
    save_model_artifacts(model, ticker, config)
    
    # Generate plots
    reports_path = Path(config['paths']['figures'])
    
    if args.mode in ['walk_forward', 'train_test_split']:
        plot_confusion_matrix(y_true, y_pred_rf, ticker,
                             str(reports_path / f"{ticker}_confusion_matrix.png"))
        plot_roc_curve(y_true, y_proba_rf, y_proba_lr, ticker,
                      str(reports_path / f"{ticker}_roc_curve.png"))
        
        # Feature importance
        feature_imp = model.get_feature_importances(top_n=20)
        plot_feature_importance(feature_imp, ticker,
                               str(reports_path / f"{ticker}_feature_importance.png"))
        
        # Save metrics
        save_metrics_summary(metrics, ticker, str(Path(config['paths']['reports']) / f"{ticker}_metrics.csv"))
    
    # Timing
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print_section(f"✅ Training Complete")
    logger.info(f"⏱️  Time elapsed: {elapsed:.2f} seconds")
    logger.info(f"📦 Model saved: models/{ticker}_predictor.joblib")
    logger.info(f"📊 Reports saved to: reports/figures/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
