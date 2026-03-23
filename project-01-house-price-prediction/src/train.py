from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.utils import ensure_dir

def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric features."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, feature_names)],
        remainder="drop"
    )
    return preprocessor

def get_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch California housing dataset."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return X, y

def main(model_type: str, out_dir: str, random_state: int) -> None:
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    feature_names = list(X_train.columns)
    preprocessor = build_preprocessor(feature_names)

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError("model_type must be one of: linear, rf")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    out_path = Path(out_dir)
    ensure_dir(out_path)

    # Save pipeline
    model_path = out_path / f"{model_type}_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    # Save test split
    split_path = out_path / "test_split.joblib"
    joblib.dump({"X_test": X_test, "y_test": y_test}, split_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved test split to: {split_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "rf"])
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args.model_type, args.out_dir, args.random_state)
