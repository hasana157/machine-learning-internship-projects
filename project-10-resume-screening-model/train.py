"""
train.py
--------
Main entry-point: runs the full training pipeline end-to-end.

Run:
    python train.py

Steps
-----
1. Load and engineer dataset (data_loader)
2. Stratified train/val split (trainer)
3. Build and fit ColumnTransformer + LogReg Pipeline (trainer)
4. Evaluate on validation set (evaluator)
5. Save model + label map (trainer)
"""

import logging

from src.data_loader import load_dataset, get_X_y
from src.trainer     import split_data, train, save_model
from src.evaluator   import evaluate
from src.utils       import section

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    section("PROJECT 10 — RESUME SCREENING MODEL  (Educational Use Only)")

    # 1. Load & engineer features
    df = load_dataset()
    X, y = get_X_y(df)

    # 2. Stratified split
    X_train, X_val, y_train, y_val = split_data(X, y)

    # 3. Train pipeline
    pipeline = train(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate(pipeline, X_val, y_val)

    # 5. Persist
    save_model(pipeline, pipeline.classes_)

    section("PIPELINE COMPLETE")
    print(f"  Macro F1   : {metrics['macro_f1']}")
    print(f"  Accuracy   : {metrics['accuracy']}")
    print(f"  Model saved: models/resume_classifier.joblib")
    print(f"  Metrics    : reports/metrics.json")
    print(f"  CM Plot    : reports/confusion_matrix.png")
    print()
    print("  [!] This model is for EDUCATIONAL purposes only.")
    print("  [!] DO NOT use for real hiring or candidate screening.")
    print()


if __name__ == "__main__":
    main()
