import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils import ensure_dir


def load_data(path="data/raw/churn_data.csv"):
    """
    Load dataset
    """
    df = pd.read_csv(path)
    return df


def build_pipeline(model_type):

    num_features = ["tenure_months", "monthly_charges"]
    cat_features = [
        "contract_type",
        "internet_service",
        "payment_method"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)

    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )

    else:
        raise ValueError("Invalid model type")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    return pipeline


def main(model_type, out_dir):

    df = load_data()

    X = df.drop(columns="churn")
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    pipeline = build_pipeline(model_type)

    pipeline.fit(X_train, y_train)

    ensure_dir(out_dir)

    joblib.dump(
        pipeline,
        f"{out_dir}/{model_type}_model.joblib"
    )

    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test
        },
        f"{out_dir}/test_split.joblib"
    )

    print(f"Model saved: {model_type}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        choices=["logreg", "rf"],
        required=True
    )

    parser.add_argument(
        "--out_dir",
        default="models"
    )

    args = parser.parse_args()

    main(args.model_type, args.out_dir)