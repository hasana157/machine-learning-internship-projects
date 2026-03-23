import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score
)

from src.utils import ensure_dir, save_json


def find_best_threshold(y_true, probs):

    thresholds = np.linspace(0.01, 0.9, 200)

    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:

        preds = probs >= t
        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def evaluate_model(model_path, reports_dir):

    model = joblib.load(model_path)
    split = joblib.load("models/test_split.joblib")

    X_test = split["X_test"]
    y_test = split["y_test"]

    probs = model.predict_proba(X_test)[:, 1]

    # PR AUC
    pr_auc = average_precision_score(y_test, probs)

    # threshold search
    best_threshold, best_f1 = find_best_threshold(y_test, probs)

    # precision recall curve
    precision, recall, _ = precision_recall_curve(y_test, probs)

    ensure_dir(f"{reports_dir}/figures")

    plt.figure()

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.savefig(
        f"{reports_dir}/figures/pr_curve.png",
        dpi=150
    )

    metrics = {
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold)
    }

    save_json(
        f"{reports_dir}/metrics.json",
        metrics
    )

    print("Evaluation Metrics")
    print(metrics)


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

    evaluate_model(
        args.model_path,
        args.reports_dir
    )