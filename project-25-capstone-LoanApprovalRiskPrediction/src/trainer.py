"""
Training orchestration: stratified train/test split, hyperparameter
search per candidate model, best-model selection by cross-validated
ROC-AUC, and optional MLflow experiment logging.
"""

import json
from pathlib import Path

import joblib
import mlflow
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.model_factory import build_candidate_pipelines
from src.utils import get_logger

logger = get_logger(__name__)


def split_data(df, config: dict):
    target_col = config["data"]["target_column"]
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    return train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=y,
    )


def tune_candidates(X_train, y_train, config: dict) -> dict:
    train_cfg = config["training"]
    cv = StratifiedKFold(n_splits=train_cfg["cv_folds"], shuffle=True,
                          random_state=config["project"]["random_state"])

    candidates = build_candidate_pipelines(config)
    results = {}

    mlflow_enabled = config["mlflow"]["enabled"]
    if mlflow_enabled:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    for name, (pipe, grid) in candidates.items():
        logger.info(f"Tuning {name} ({train_cfg['n_iter_search']} search iterations, "
                     f"{train_cfg['cv_folds']}-fold CV)...")

        search = RandomizedSearchCV(
            pipe,
            param_distributions=grid,
            n_iter=train_cfg["n_iter_search"],
            scoring=train_cfg["scoring"],
            cv=cv,
            random_state=config["project"]["random_state"],
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        logger.info(f"  {name}: best CV {train_cfg['scoring']}={search.best_score_:.4f} "
                     f"| params={search.best_params_}")

        results[name] = {
            "best_estimator": search.best_estimator_,
            "best_cv_score": search.best_score_,
            "best_params": search.best_params_,
        }

        if mlflow_enabled:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(search.best_params_)
                mlflow.log_metric(f"cv_{train_cfg['scoring']}", search.best_score_)
                mlflow.set_tag("model_family", name)

    return results


def select_best(results: dict):
    best_name = max(results, key=lambda k: results[k]["best_cv_score"])
    return best_name, results[best_name]


def save_pipeline(pipeline, config: dict):
    out_path = Path(config["paths"]["best_pipeline"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    logger.info(f"Saved production pipeline -> {out_path}")


def save_training_summary(results: dict, best_name: str, config: dict):
    summary = {
        "best_model": best_name,
        "candidates": {
            name: {"cv_score": r["best_cv_score"], "best_params": r["best_params"]}
            for name, r in results.items()
        },
    }
    out_path = Path(config["paths"]["artifacts_dir"]) / "training_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Training summary -> {out_path}")
