# src/evaluate.py
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from src.utils import ensure_dir, save_json

def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = [(t, f1_score(y_true, probs >= t)) for t in thresholds]
    return max(scores, key=lambda x: x[1])

def main(model_path, reports_dir):
    model = joblib.load(model_path)
    split = joblib.load("models/test_split.joblib")
    X_test = split["X_test"]
    y_test = split["y_test"]

    # Predict probabilities
    probs = model.predict_proba(X_test)[:,1]

    # Threshold tuning
    best_threshold, best_f1 = find_best_threshold(y_test, probs)
    preds = (probs >= best_threshold).astype(int)

    # Metrics
    metrics = {
        "f1": float(best_f1),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "best_threshold": float(best_threshold)
    }
    print("Metrics:", metrics)

    # Save metrics
    ensure_dir(reports_dir)
    save_json(f"{reports_dir}/metrics.json", metrics)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{reports_dir}/figures/confusion_matrix.png", dpi=150)
    plt.show()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, probs)
    plt.title("ROC Curve")
    plt.savefig(f"{reports_dir}/figures/roc_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()
    main(args.model_path, args.reports_dir)
