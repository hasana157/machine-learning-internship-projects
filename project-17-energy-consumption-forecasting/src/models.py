"""
Candidate model registry.

Every model in config.yaml -> model -> candidates gets trained and
evaluated; the one with the lowest test MAE is saved as the
production model (models/best_model.joblib).
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def build_candidates(config: dict) -> dict:
    cfg = config["model"]
    candidates = cfg["candidates"]
    random_state = cfg["random_state"]

    models = {}

    if "linear_regression" in candidates:
        models["linear_regression"] = LinearRegression()

    if "random_forest" in candidates:
        params = candidates["random_forest"]
        models["random_forest"] = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            random_state=random_state,
        )

    if "gradient_boosting" in candidates:
        params = candidates["gradient_boosting"]
        models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            random_state=random_state,
        )

    return models
