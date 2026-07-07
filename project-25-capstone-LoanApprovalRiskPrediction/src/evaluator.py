"""
Business-focused evaluation for a credit risk model.

Accuracy alone is a poor fit for loan approval: classes are imbalanced
and a wrong probability is costlier than a wrong label (a rejected
applicant who was actually low-risk is a lost customer; an approved
high-risk applicant is a potential default). So this module reports
ROC-AUC, PR-AUC, F1, and a calibration curve alongside the confusion
matrix, and saves all of it as both plots and a JSON metrics file.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay, average_precision_score,
    brier_score_loss, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_test, y_pred, y_proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def plot_roc_curves(fitted_models: dict, X_test, y_test, figures_dir: str):
    plt.figure(figsize=(6.5, 5.5))
    ax = plt.gca()
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curve — Model Comparison")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/roc_curve.png", dpi=150)
    plt.close()


def plot_precision_recall(best_model, X_test, y_test, figures_dir: str):
    plt.figure(figsize=(6.5, 5.5))
    PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("Precision-Recall Curve — Best Model")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/precision_recall_curve.png", dpi=150)
    plt.close()


def plot_calibration(best_model, X_test, y_test, figures_dir: str):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)

    plt.figure(figsize=(6, 5.5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve — Best Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/calibration_curve.png", dpi=150)
    plt.close()


def plot_feature_importance(best_model, best_name: str, feature_names: list,
                             figures_dir: str, artifacts_dir: str):
    classifier = best_model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning(f"{best_name} exposes no importances/coefficients; skipping chart.")
        return None

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(f"{artifacts_dir}/feature_importance.csv", index=False)

    top = imp_df.head(15)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1], color="#1D3557")
    plt.title(f"Top 15 Feature Importances — {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/feature_importance.png", dpi=150)
    plt.close()

    return imp_df


def write_metrics_report(metrics_by_model: dict, best_name: str, metrics_path: str):
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"best_model": best_name, "metrics": metrics_by_model}, f, indent=2)
    logger.info(f"Metrics saved -> {metrics_path}")
