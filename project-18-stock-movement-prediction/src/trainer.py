"""
Training module with walk-forward validation to prevent lookahead bias.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .model import StockMovementPredictor
from .utils import save_model_metadata

logger = logging.getLogger(__name__)


def walk_forward_validation(
    df: pd.DataFrame,
    model: StockMovementPredictor,
    config: Dict
) -> pd.DataFrame:
    """Perform walk-forward validation WITHOUT lookahead bias.
    
    CRITICAL: Features at time t only use data from times <= t
    
    Args:
        df: DataFrame with features and target (features.engineer_features() output)
        model: StockMovementPredictor instance
        config: Configuration dictionary
        
    Returns:
        DataFrame with predictions and actuals
    """
    
    wf_config = config['training']['walk_forward']
    initial_window = wf_config['initial_window']
    retrain_frequency = wf_config['retrain_frequency']
    
    predictions_list = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 Starting Walk-Forward Validation")
    logger.info(f"   Initial window: {initial_window} days")
    logger.info(f"   Retrain frequency: {retrain_frequency} days")
    logger.info(f"   Total data: {len(df)} days")
    logger.info(f"{'='*60}\n")
    
    # Walk forward loop
    for i in range(initial_window, len(df), 1):
        
        # Retrain at specified intervals
        if i == initial_window or (i - initial_window) % retrain_frequency == 0:
            # Use all data up to time i (expanding window, NO lookahead)
            train_data = df.iloc[:i]
            
            X_train = train_data.drop(columns=['target'])
            y_train = train_data['target']
            
            # Fit model
            model.fit(X_train, y_train)
            
            logger.info(f"📚 Retrained at day {i}/{len(df)}: {train_data.index[-1].date()}")
        
        # Predict for day i
        test_data = df.iloc[i:i+1]
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # Get predictions with confidence
        pred, confidence = model.predict_with_confidence(X_test)
        proba = model.predict_proba(X_test)[:, 1]  # Probability of UP
        
        pred_baseline = model.predict_baseline(X_test)
        proba_baseline = model.predict_proba_baseline(X_test)[:, 1]
        
        # Correct/Incorrect
        correct = (pred[0] == y_test.values[0])
        correct_baseline = (pred_baseline[0] == y_test.values[0])
        
        predictions_list.append({
            'date': test_data.index[0],
            'close_price': test_data['close'].values[0],
            'prediction_rf': pred[0],
            'actual': y_test.values[0],
            'correct_rf': int(correct),
            'probability_up_rf': proba[0],
            'confidence_rf': confidence[0],
            'prediction_lr': pred_baseline[0],
            'correct_lr': int(correct_baseline),
            'probability_up_lr': proba_baseline[0]
        })
    
    results_df = pd.DataFrame(predictions_list)
    
    # Calculate metrics
    accuracy_rf = accuracy_score(results_df['actual'], results_df['prediction_rf'])
    accuracy_lr = accuracy_score(results_df['actual'], results_df['prediction_lr'])
    
    precision_rf = precision_score(results_df['actual'], results_df['prediction_rf'], zero_division=0)
    precision_lr = precision_score(results_df['actual'], results_df['prediction_lr'], zero_division=0)
    
    recall_rf = recall_score(results_df['actual'], results_df['prediction_rf'], zero_division=0)
    recall_lr = recall_score(results_df['actual'], results_df['prediction_lr'], zero_division=0)
    
    f1_rf = f1_score(results_df['actual'], results_df['prediction_rf'], zero_division=0)
    f1_lr = f1_score(results_df['actual'], results_df['prediction_lr'], zero_division=0)
    
    roc_auc_rf = roc_auc_score(results_df['actual'], results_df['probability_up_rf'])
    roc_auc_lr = roc_auc_score(results_df['actual'], results_df['probability_up_lr'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Walk-Forward Validation Results")
    logger.info(f"{'='*60}")
    logger.info(f"\nRandom Forest:")
    logger.info(f"  Accuracy:  {accuracy_rf:.4f}")
    logger.info(f"  Precision: {precision_rf:.4f}")
    logger.info(f"  Recall:    {recall_rf:.4f}")
    logger.info(f"  F1-Score:  {f1_rf:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc_rf:.4f}")
    
    logger.info(f"\nLogistic Regression:")
    logger.info(f"  Accuracy:  {accuracy_lr:.4f}")
    logger.info(f"  Precision: {precision_lr:.4f}")
    logger.info(f"  Recall:    {recall_lr:.4f}")
    logger.info(f"  F1-Score:  {f1_lr:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc_lr:.4f}")
    
    logger.info(f"\nClass Distribution:")
    logger.info(f"  UP days: {int(results_df['actual'].sum())}")
    logger.info(f"  DOWN days: {len(results_df) - int(results_df['actual'].sum())}")
    logger.info(f"{'='*60}\n")
    
    return results_df


def train_test_split_validation(
    df: pd.DataFrame,
    model: StockMovementPredictor,
    config: Dict
) -> Dict:
    """Train-test split validation (chronological split, NO shuffle).
    
    Args:
        df: DataFrame with features and target
        model: StockMovementPredictor instance
        config: Configuration dictionary
        
    Returns:
        Dictionary with metrics
    """
    
    test_size = config['training']['train_test_split']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_rf = model.predict(X_test)
    y_proba_rf = model.predict_proba(X_test)[:, 1]
    
    y_pred_lr = model.predict_baseline(X_test)
    y_proba_lr = model.predict_proba_baseline(X_test)[:, 1]
    
    metrics = {
        'random_forest': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf, zero_division=0),
            'recall': recall_score(y_test, y_pred_rf, zero_division=0),
            'f1': f1_score(y_test, y_pred_rf, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_rf)
        },
        'logistic_regression': {
            'accuracy': accuracy_score(y_test, y_pred_lr),
            'precision': precision_score(y_test, y_pred_lr, zero_division=0),
            'recall': recall_score(y_test, y_pred_lr, zero_division=0),
            'f1': f1_score(y_test, y_pred_lr, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_lr)
        }
    }
    
    logger.info(f"✅ Train-test split: {len(train_data)} train, {len(test_data)} test")
    logger.info(f"   RF Accuracy: {metrics['random_forest']['accuracy']:.4f}")
    logger.info(f"   LR Accuracy: {metrics['logistic_regression']['accuracy']:.4f}")
    
    return metrics


def save_model_artifacts(
    model: StockMovementPredictor,
    ticker: str,
    config: Dict
) -> None:
    """Save model and scaler artifacts.
    
    Args:
        model: Trained model
        ticker: Ticker symbol
        config: Configuration dictionary
    """
    models_path = Path(config['paths']['models'])
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_path / f"{ticker}_predictor.joblib"
    model.save(str(model_path))
    
    # Save metadata
    metadata = model.get_model_metadata()
    save_model_metadata(ticker, metadata, config)
    
    logger.info(f"✅ Saved model artifacts for {ticker}")


def load_model_artifacts(
    ticker: str,
    config: Dict
) -> StockMovementPredictor:
    """Load model and metadata.
    
    Args:
        ticker: Ticker symbol
        config: Configuration dictionary
        
    Returns:
        Loaded model
    """
    models_path = Path(config['paths']['models'])
    model_path = models_path / f"{ticker}_predictor.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for {ticker} at {model_path}")
    
    model = StockMovementPredictor.load(str(model_path))
    return model
