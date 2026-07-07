"""
Train a Cat vs Dog classifier via transfer learning:
  Phase 1 — freeze MobileNetV2 backbone, train the classification head
  Phase 2 — unfreeze top backbone layers, fine-tune at a low learning rate

Usage:
    python -m src.train
"""

import json

import matplotlib.pyplot as plt

from src.config import (
    FINE_TUNE_EPOCHS,
    HEAD_EPOCHS,
    METRICS_PATH,
    MODEL_PATH,
    TRAINING_CURVE_PATH,
)
from src.data import load_datasets
from src.model import build_model, enable_fine_tuning


def merge_histories(history1, history2):
    """Concatenate phase-1 and phase-2 History.history dicts for one combined plot."""
    combined = {k: list(v) for k, v in history1.history.items()}
    for k in history2.history:
        combined[k].extend(history2.history[k])
    return combined


def plot_history(history_dict, out_path, phase1_epochs):
    plt.figure(figsize=(7, 5))
    plt.plot(history_dict["accuracy"], label="train_acc")
    plt.plot(history_dict["val_accuracy"], label="val_acc")
    plt.axvline(
        x=phase1_epochs - 0.5,
        color="gray",
        linestyle="--",
        label="fine-tuning starts",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([min(min(history_dict["accuracy"]), min(history_dict["val_accuracy"])) - 0.05, 1.0])
    plt.legend(loc="lower right")
    plt.title("Training Curve — Cat vs Dog Transfer Learning")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curve to {out_path}")


def main():
    print("Loading Cats vs Dogs dataset (downloads on first run)...")
    train_ds, val_ds, class_names = load_datasets()
    print("Class names:", class_names)

    print("Building model (MobileNetV2 backbone, frozen)...")
    model, base_model = build_model(weights="imagenet")
    model.summary()

    print(f"\n--- Phase 1: training classification head ({HEAD_EPOCHS} epochs) ---")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS)

    print(f"\n--- Phase 2: fine-tuning top backbone layers ({FINE_TUNE_EPOCHS} epochs) ---")
    enable_fine_tuning(model, base_model)
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS)

    print("\nEvaluating final model on validation set...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    combined_history = merge_histories(history1, history2)
    plot_history(combined_history, TRAINING_CURVE_PATH, phase1_epochs=HEAD_EPOCHS)

    metrics = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "head_epochs": HEAD_EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "class_names": class_names,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
