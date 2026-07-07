"""
Train a neural network on MNIST and save the trained model to disk.

Usage:
    python -m src.train
"""

import json

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import EPOCHS, MODEL_PATH, REPORTS_DIR, VALIDATION_SPLIT
from src.model import build_model


def load_data():
    """Load and normalize the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


def plot_history(history):
    """Save accuracy/loss curves to reports/ for a quick visual sanity check."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path = REPORTS_DIR / "training_history.png"
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    print("Loading MNIST data...")
    x_train, y_train, x_test, y_test = load_data()

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training...")
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save model
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Save training plots
    plot_history(history)

    # Save a small metrics report
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs": EPOCHS,
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
