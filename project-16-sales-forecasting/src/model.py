"""Model construction — a thin factory over scikit-learn estimators.

Keeping this separate from trainer.py means swapping in a new model
type (e.g. XGBoost, GradientBoosting) only touches this one file.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

VALID_MODEL_TYPES = ("linear", "rf")


def build_model(model_type: str, config: dict):
    if model_type == "linear":
        params = config["model"]["linear"]
        return LinearRegression(**params)

    if model_type == "rf":
        params = config["model"]["random_forest"]
        return RandomForestRegressor(**params)

    raise ValueError(
        f"Unknown model_type '{model_type}'. Choose one of {VALID_MODEL_TYPES}."
    )
