import argparse
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data import load_dataset
from src.utils import ensure_dir


def train_model(strategy, out_dir):

    X, y = load_dataset()

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    # ----- Strategy 1 : Class weighting -----
    if strategy == "class_weight":

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1
        )

        X_train_final = X_train
        y_train_final = y_train

    # ----- Strategy 2 : Undersampling -----
    elif strategy == "undersample":

        fraud = y_train == 1
        legit = y_train == 0

        fraud_idx = y_train[fraud].index

        legit_idx = y_train[legit].sample(
            n=len(fraud_idx) * 5,
            random_state=42
        ).index

        selected_idx = fraud_idx.union(legit_idx)

        X_train_final = X_train.loc[selected_idx]
        y_train_final = y_train.loc[selected_idx]

        model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )

    else:
        raise ValueError("Strategy must be 'class_weight' or 'undersample'")

    # pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train_final, y_train_final)

    ensure_dir(out_dir)

    # save model
    joblib.dump(
        pipeline,
        f"{out_dir}/{strategy}_model.joblib"
    )

    # save test split for evaluation
    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test
        },
        f"{out_dir}/test_split.joblib"
    )

    print(f"Model saved: {strategy}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        choices=["class_weight", "undersample"],
        required=True
    )

    parser.add_argument(
        "--out_dir",
        default="models"
    )

    args = parser.parse_args()

    train_model(args.strategy, args.out_dir)