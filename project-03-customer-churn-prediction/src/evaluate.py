import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from src.utils import ensure_dir, save_json


def main(model_path, reports_dir):

    pipeline = joblib.load(model_path)

    split = joblib.load("models/test_split.joblib")

    X_test = split["X_test"]
    y_test = split["y_test"]

    preds = pipeline.predict(X_test)

    f1 = f1_score(y_test, preds)

    print("F1 Score:", f1)

    r = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1"
    )

    feature_names = X_test.columns

    importance = pd.Series(
    r.importances_mean,
    index=feature_names
    ).sort_values(ascending=False)

    ensure_dir(f"{reports_dir}/figures")

    importance.head(10).plot(
        kind="barh",
        title="Top Churn Drivers"
    )

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        f"{reports_dir}/figures/churn_drivers.png",
        dpi=150
    )

    importance.head(10).to_csv(
        f"{reports_dir}/top_churn_drivers.csv"
    )

    save_json(
        f"{reports_dir}/metrics.json",
        {"f1_score": float(f1)}
    )

    print("\nTop churn drivers:")
    print(importance.head(10))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        required=True
    )

    parser.add_argument(
        "--reports_dir",
        default="reports"
    )

    args = parser.parse_args()

    main(args.model_path, args.reports_dir)