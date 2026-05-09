"""
evaluator.py
------------
Computes and persists classification metrics.

Primary metric: macro F1-score (most robust for multi-class / imbalanced data).
Additional metrics: accuracy, per-class precision, recall, F1.
Outputs: console report, JSON file, confusion-matrix PNG.
"""

import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import METRICS_PATH, REPORT_DIR
from src.utils import ensure_dirs, save_json, section

logger = logging.getLogger(__name__)
CM_PLOT_PATH = os.path.join(REPORT_DIR, "confusion_matrix.png")


def evaluate(pipeline, X_val, y_val) -> Dict:
    """
    Run evaluation on the validation set and persist results.

    Returns dict with summary and per-class metrics.
    """
    y_pred = pipeline.predict(X_val)
    labels = sorted(y_val.unique())

    accuracy   = accuracy_score(y_val, y_pred)
    macro_f1   = f1_score(y_val, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_val, y_pred, average="macro", zero_division=0)

    report_dict = classification_report(
        y_val, y_pred, labels=labels, output_dict=True, zero_division=0,
    )
    per_class = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1_score":  round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in labels if cls in report_dict
    }

    metrics = {
        "accuracy":        round(accuracy,   4),
        "macro_f1":        round(macro_f1,   4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall":    round(macro_rec,  4),
        "per_class":       per_class,
        "num_val_samples": int(len(y_val)),
    }

    section("EVALUATION RESULTS")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Macro F1-Score  : {macro_f1:.4f}  <- primary metric")
    print(f"  Macro Precision : {macro_prec:.4f}")
    print(f"  Macro Recall    : {macro_rec:.4f}")
    print()
    print(classification_report(y_val, y_pred, labels=labels, zero_division=0))

    ensure_dirs(REPORT_DIR)
    save_json(metrics, METRICS_PATH)
    _plot_confusion_matrix(y_val, y_pred, labels)
    return metrics


def _plot_confusion_matrix(y_true, y_pred, labels) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(
        "Confusion Matrix — Resume Classifier\n(Educational Use Only — NOT a hiring tool)",
        fontsize=11,
    )
    plt.tight_layout()
    ensure_dirs(REPORT_DIR)
    fig.savefig(CM_PLOT_PATH, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", CM_PLOT_PATH)
