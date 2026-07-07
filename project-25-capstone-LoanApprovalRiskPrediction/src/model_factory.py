"""
Candidate model registry + hyperparameter search spaces.

Every enabled model in config.yaml -> models gets wrapped in a full
Pipeline (preprocessing + estimator) and tuned with RandomizedSearchCV
on ROC-AUC. This keeps preprocessing INSIDE cross-validation so there's
no leakage from fitting the scaler/imputer on the full training set.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def build_candidate_pipelines(config: dict) -> dict:
    preprocessor = build_preprocessor(config)
    models_cfg = config["models"]
    random_state = config["project"]["random_state"]

    candidates = {}

    if models_cfg["logistic_regression"]["enabled"]:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
        ])
        grid = {f"classifier__{k}": v for k, v in models_cfg["logistic_regression"]["param_grid"].items()}
        candidates["logistic_regression"] = (pipe, grid)

    if models_cfg["random_forest"]["enabled"]:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ])
        grid = {f"classifier__{k}": v for k, v in models_cfg["random_forest"]["param_grid"].items()}
        candidates["random_forest"] = (pipe, grid)

    if models_cfg["xgboost"]["enabled"]:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ])
        grid = {f"classifier__{k}": v for k, v in models_cfg["xgboost"]["param_grid"].items()}
        candidates["xgboost"] = (pipe, grid)

    return candidates
