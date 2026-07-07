"""
Explainability layer (SHAP).

Two use cases:
  1. `save_global_summary` -- a summary bar chart used in the training
     report, showing which features drive risk across the whole test set.
  2. `explain_single_prediction` -- per-applicant top contributing
     features, used by predict.py to justify an individual decision
     (a hard requirement for real banking risk systems, not a nice-to-have).

Falls back gracefully if SHAP isn't installed / a model type isn't
supported by the fast TreeExplainer path, so the rest of the pipeline
never breaks because of this module.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _build_explainer(classifier, background_data):
    model_type = type(classifier).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        return shap.TreeExplainer(classifier)
    return shap.LinearExplainer(classifier, background_data)


def save_background_sample(pipeline, X_train, background_path: str, sample_size: int = 100):
    """Persist a transformed sample of the TRAINING data to disk. Single-row
    predictions need a real reference distribution to compare against --
    comparing a row to itself always yields zero attribution."""
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_train.sample(
        n=min(sample_size, len(X_train)), random_state=42))
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    Path(background_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_transformed, background_path)
    logger.info(f"SHAP background sample saved -> {background_path}")


def _load_background(background_path: str):
    if Path(background_path).exists():
        return joblib.load(background_path)
    return None


def save_global_summary(pipeline, X_test, feature_names: list, figures_dir: str, sample_size: int = 300):
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed -- skipping global explanation plot.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    sample = X_test_transformed[:sample_size]

    try:
        explainer = _build_explainer(classifier, sample)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class for tree models

        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=feature_names,
                           show=False, plot_size=(9, 6))
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved -> {figures_dir}/shap_summary.png")
    except Exception as e:
        logger.warning(f"SHAP summary plot failed ({e}); continuing without it.")


def explain_single_prediction(pipeline, X_row: pd.DataFrame, feature_names: list,
                               top_n: int = 5, background_path: str = "artifacts/shap_background.joblib") -> list:
    """Returns top_n (feature, shap_value) pairs for one applicant row,
    measured against a saved training-data background sample."""
    if not SHAP_AVAILABLE:
        return []

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    background = _load_background(background_path)
    if background is None:
        logger.warning("No saved SHAP background found -- explanation quality will be poor. "
                        "Re-run training to generate artifacts/shap_background.joblib.")
        background = X_transformed

    try:
        explainer = _build_explainer(classifier, background)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        row_values = shap_values[0]
        order = np.argsort(-np.abs(row_values))[:top_n]
        return [(feature_names[i], float(row_values[i])) for i in order]
    except Exception as e:
        logger.warning(f"Per-prediction SHAP explanation failed ({e}).")
        return []
