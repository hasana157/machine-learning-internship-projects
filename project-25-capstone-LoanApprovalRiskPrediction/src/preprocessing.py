"""
Preprocessing: builds the sklearn ColumnTransformer used inside every
model Pipeline. Keeping this as one shared function means every
candidate model sees an identical, leakage-safe preprocessing step
(imputation + scaling fit only on training folds via Pipeline/CV).
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(config: dict) -> ColumnTransformer:
    cols = config["columns"]
    numeric_cols = cols["numeric"] + cols["engineered"]
    categorical_cols = cols["categorical"]
    prep_cfg = config["preprocessing"]

    numeric_steps = [("imputer", SimpleImputer(strategy=prep_cfg["numeric_imputer_strategy"]))]
    if prep_cfg["scale_numeric"]:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=prep_cfg["categorical_imputer_strategy"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Flat, human-readable feature names after the ColumnTransformer
    (used for feature importance tables and SHAP labeling)."""
    return list(preprocessor.get_feature_names_out())
