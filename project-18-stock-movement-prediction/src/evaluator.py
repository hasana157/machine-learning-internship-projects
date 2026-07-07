"""
Model evaluation and visualization module.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, matthews_corrcoef
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


def generate_classification_report(
    y_true: np.ndarray,
    y_pred_rf: np.ndarray,
    y_proba_rf: np.ndarray,
    y_pred_lr: np.ndarray = None,
    y_proba_lr: np.ndarray = None,
    ticker: str = "UNKNOWN",
    save_path: str = None
) -> dict:
    """Generate classification metrics.
    
    Args:
        y_true: True labels
        y_pred_rf: RF predictions
        y_proba_rf: RF probabilities
        y_pred_lr: LR predictions
        y_proba_lr: LR probabilities
        ticker: Ticker symbol
        save_path: Path to save report
        
    Returns:
        Metrics dictionary
    """
    
    metrics_rf = {
        'accuracy': accuracy_score(y_true, y_pred_rf),
        'precision': precision_score(y_true, y_pred_rf, zero_division=0),
        'recall': recall_score(y_true, y_pred_rf, zero_division=0),
        'f1': f1_score(y_true, y_pred_rf, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba_rf),
        'mcc': matthews_corrcoef(y_true, y_pred_rf)
    }
    
    results = {'random_forest': metrics_rf}
    
    if y_pred_lr is not None and y_proba_lr is not None:
        metrics_lr = {
            'accuracy': accuracy_score(y_true, y_pred_lr),
            'precision': precision_score(y_true, y_pred_lr, zero_division=0),
            'recall': recall_score(y_true, y_pred_lr, zero_division=0),
            'f1': f1_score(y_true, y_pred_lr, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba_lr),
            'mcc': matthews_corrcoef(y_true, y_pred_lr)
        }
        results['logistic_regression'] = metrics_lr
    
    # Print report
    print(f"\n{'='*60}")
    print(f"📊 Classification Metrics for {ticker}")
    print(f"{'='*60}")
    print(f"\n{'Metric':<20} {'Random Forest':<15} {'Logistic Reg':<15}")
    print("-" * 60)
    for metric in metrics_rf.keys():
        rf_val = metrics_rf[metric]
        lr_val = metrics_lr[metric] if y_pred_lr is not None else None
        if lr_val is not None:
            print(f"{metric:<20} {rf_val:<15.4f} {lr_val:<15.4f}")
        else:
            print(f"{metric:<20} {rf_val:<15.4f}")
    print(f"{'='*60}\n")
    
    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = "UNKNOWN",
    save_path: str = None,
    title: str = "Confusion Matrix"
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        ticker: Ticker symbol
        save_path: Path to save figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title(f"{title} - {ticker}")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba_rf: np.ndarray,
    y_proba_lr: np.ndarray = None,
    ticker: str = "UNKNOWN",
    save_path: str = None
) -> None:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba_rf: RF probabilities
        y_proba_lr: LR probabilities
        ticker: Ticker symbol
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # RF ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_true, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2)
    
    # LR ROC
    if y_proba_lr is not None:
        fpr_lr, tpr_lr, _ = roc_curve(y_true, y_proba_lr)
        auc_lr = auc(fpr_lr, tpr_lr)
        plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})', linewidth=2)
    
    # Diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {ticker}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved ROC curve to {save_path}")
    
    plt.close()


def plot_feature_importance(
    feature_importance: pd.Series,
    ticker: str = "UNKNOWN",
    save_path: str = None,
    top_n: int = 20
) -> None:
    """Plot feature importance.
    
    Args:
        feature_importance: Feature importance Series
        ticker: Ticker symbol
        save_path: Path to save figure
        top_n: Number of top features to plot
    """
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    top_features.sort_values().plot(kind='barh', color='steelblue')
    plt.title(f'Top {top_n} Feature Importance - {ticker}')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved feature importance to {save_path}")
    
    plt.close()


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series = None,
    ticker: str = "UNKNOWN",
    save_path: str = None
) -> None:
    """Plot equity curve vs benchmark.
    
    Args:
        equity_curve: Strategy equity Series
        benchmark_curve: Benchmark (buy-and-hold) equity Series
        ticker: Ticker symbol
        save_path: Path to save figure
    """
    plt.figure(figsize=(14, 7))
    
    # Normalize to percentage
    equity_normalized = (equity_curve / equity_curve.iloc[0] - 1) * 100
    
    plt.plot(equity_normalized.index, equity_normalized.values, 
             label='Strategy', linewidth=2, color='green')
    
    if benchmark_curve is not None:
        benchmark_normalized = (benchmark_curve / benchmark_curve.iloc[0] - 1) * 100
        plt.plot(benchmark_normalized.index, benchmark_normalized.values,
                 label='Buy & Hold', linewidth=2, color='blue', alpha=0.7)
        
        # Fill area
        plt.fill_between(equity_normalized.index, 
                        equity_normalized.values, 
                        benchmark_normalized.values,
                        alpha=0.2, color='gray')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.title(f'Equity Curve - {ticker}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved equity curve to {save_path}")
    
    plt.close()


def plot_drawdown(
    equity_curve: pd.Series,
    ticker: str = "UNKNOWN",
    save_path: str = None
) -> None:
    """Plot drawdown chart (underwater plot).
    
    Args:
        equity_curve: Equity Series
        ticker: Ticker symbol
        save_path: Path to save figure
    """
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100
    
    plt.figure(figsize=(14, 6))
    plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.6)
    plt.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.title(f'Underwater Plot - {ticker}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved drawdown chart to {save_path}")
    
    plt.close()


def save_metrics_summary(
    metrics: dict,
    ticker: str,
    save_path: str = None
) -> None:
    """Save metrics summary to CSV.
    
    Args:
        metrics: Metrics dictionary
        ticker: Ticker symbol
        save_path: Path to save CSV
    """
    if save_path is None:
        save_path = f"reports/{ticker}_metrics.csv"
    
    # Flatten metrics dict
    flat_metrics = {}
    for model_name, model_metrics in metrics.items():
        for metric_name, value in model_metrics.items():
            flat_metrics[f"{model_name}_{metric_name}"] = value
    
    metrics_df = pd.DataFrame([flat_metrics])
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(save_path, index=False)
    logger.info(f"✅ Saved metrics summary to {save_path}")
