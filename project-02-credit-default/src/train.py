# src/train.py
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.utils import ensure_dir

DATA_PATH = "data/creditcard.csv"
TARGET_COLUMN = "Class"  # Update according to your dataset

def load_data():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} not found in dataset")
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y

def main(model_type, out_dir):
    X, y = load_data()

    # Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Choose model
    if model_type == "logreg":
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif model_type == "gb":
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    else:
        raise ValueError("Invalid model type")

    # Pipeline: scale features then train
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Save model and test split
    ensure_dir(out_dir)
    joblib.dump(pipeline, f"{out_dir}/{model_type}_model.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, f"{out_dir}/test_split.joblib")

    print(f"{model_type} model trained and saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["logreg", "gb"], required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    main(args.model_type, args.out_dir)
