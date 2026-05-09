"""
evaluate.py
-----------
Comprehensive model evaluation:
    - Test accuracy
    - Per-class precision / recall / F1  (classification_report)
    - Confusion matrix
    - Identification of weak classes + root-cause commentary

All artefacts are saved to reports/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src.config import CIFAR10_CLASSES, FIGURES_DIR


# ── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Run full evaluation: loss/accuracy, per-class metrics, confusion matrix.

    Args:
        model:      Trained Keras model.
        test_ds:    Preprocessed test tf.data.Dataset.
        x_test:     Raw (uint8) test images for visualisation.
        y_test:     Integer ground-truth labels.
        model_name: Label used in saved figure filenames.

    Returns:
        Dictionary with keys: loss, accuracy, y_pred, report_dict.
    """
    print(f"\n{'─'*50}")
    print(f"  Evaluating: {model_name}")
    print(f"{'─'*50}")

    # ── Overall loss & accuracy ──────────────────────────────────────────────
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)\n")

    # ── Per-sample predictions ───────────────────────────────────────────────
    y_pred_proba = model.predict(test_ds, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    # ── Classification report ────────────────────────────────────────────────
    print("  Classification Report:")
    print("  " + "-"*48)
    report_str  = classification_report(y_test, y_pred, target_names=CIFAR10_CLASSES)
    report_dict = classification_report(
        y_test, y_pred, target_names=CIFAR10_CLASSES, output_dict=True
    )
    # Indent each line for cleaner console output
    for line in report_str.splitlines():
        print("  " + line)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred, model_name)

    # ── Weak class analysis ──────────────────────────────────────────────────
    _analyse_weak_classes(report_dict)

    return {
        "loss":        loss,
        "accuracy":    acc,
        "y_pred":      y_pred,
        "y_pred_proba": y_pred_proba,
        "report_dict": report_dict,
    }


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}\n(row-normalised)", fontsize=14)
    plt.tight_layout()

    save_path = FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  [Saved] Confusion matrix → {save_path}")


# ── Weak class analysis ───────────────────────────────────────────────────────

def _analyse_weak_classes(report_dict: dict, threshold: float = 0.80) -> None:
    """
    Print a brief analysis of any class whose F1-score falls below `threshold`.

    Known CIFAR-10 hard classes and likely reasons:
        cat   — visually similar to dog; fur patterns overlap.
        dog   — same as above; pose variation adds difficulty.
        deer  — overlaps with horse in shape; background variation.
        truck — visually similar to automobile; shared features.
    """
    KNOWN_CONFUSIONS = {
        "cat":        "Commonly confused with 'dog' due to similar fur textures and body proportions.",
        "dog":        "Frequently misclassified as 'cat'; body shape and fur are highly similar at 32×32.",
        "deer":       "Horse-like silhouette; background/pose variation hurts generalisation.",
        "truck":      "Shares rectangular shapes and wheels with 'automobile'.",
        "automobile": "Some overlap with 'truck' class.",
        "bird":       "Small wingspan at 32×32 resolution loses distinguishing feather detail.",
    }

    weak = {
        cls: metrics["f1-score"]
        for cls, metrics in report_dict.items()
        if isinstance(metrics, dict) and metrics.get("f1-score", 1.0) < threshold
    }

    if not weak:
        print(f"\n  All classes exceed F1 ≥ {threshold:.0%} — strong overall performance.")
        return

    print(f"\n  ┌─ Weak Classes (F1 < {threshold:.0%}) " + "─"*28)
    for cls, f1 in sorted(weak.items(), key=lambda x: x[1]):
        note = KNOWN_CONFUSIONS.get(cls, "Review confusion matrix for specific error pattern.")
        print(f"  │  {cls:<12} F1={f1:.3f}  →  {note}")
    print("  └" + "─"*50)


# ── Training curve plots ──────────────────────────────────────────────────────

def plot_training_curves(
    history,
    model_name: str = "model",
) -> None:
    """
    Save accuracy and loss curves for a single training history.
    Handles both single-phase histories and merged two-phase histories.
    """
    hist = history.history if hasattr(history, "history") else history

    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, hist["accuracy"],     label="Train", linewidth=2)
    axes[0].plot(epochs, hist["val_accuracy"], label="Val",   linewidth=2, linestyle="--")
    axes[0].set_title(f"Accuracy — {model_name}", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, hist["loss"],     label="Train", linewidth=2)
    axes[1].plot(epochs, hist["val_loss"], label="Val",   linewidth=2, linestyle="--")
    axes[1].set_title(f"Loss — {model_name}", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / f"training_curves_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Training curves → {save_path}")


def plot_combined_curves(history_baseline, history_transfer) -> None:
    """Overlay both models' validation accuracy for direct comparison."""
    h_b = history_baseline.history if hasattr(history_baseline, "history") else history_baseline
    h_t = history_transfer.history  if hasattr(history_transfer,  "history") else history_transfer

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h_b["val_accuracy"], label="Baseline CNN",       linewidth=2)
    ax.plot(h_t["val_accuracy"], label="Transfer MobileNetV2", linewidth=2, linestyle="--")
    ax.set_title("Validation Accuracy Comparison", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_path = FIGURES_DIR / "comparison_val_accuracy.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Comparison plot → {save_path}")


# ── Sample prediction grid ────────────────────────────────────────────────────

def plot_sample_predictions(
    model: keras.Model,
    x_sample: np.ndarray,      # preprocessed images (N, H, W, 3)
    y_true:   np.ndarray,      # integer labels
    x_display: np.ndarray,     # raw uint8 images for display
    model_name: str = "model",
    n: int = 25,
) -> None:
    """
    Display a 5×5 grid of predictions.  Green title = correct, red = wrong.
    """
    proba  = model.predict(x_sample[:n], verbose=0)
    y_pred = np.argmax(proba, axis=1)

    fig, axes = plt.subplots(5, 5, figsize=(13, 13))
    fig.suptitle(f"Sample Predictions — {model_name}", fontsize=15, y=1.01)

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_display[i])
        ax.axis("off")
        true_lbl = CIFAR10_CLASSES[y_true[i]]
        pred_lbl = CIFAR10_CLASSES[y_pred[i]]
        conf     = proba[i, y_pred[i]] * 100
        colour   = "green" if y_pred[i] == y_true[i] else "red"
        ax.set_title(
            f"T:{true_lbl}\nP:{pred_lbl} ({conf:.0f}%)",
            color=colour, fontsize=8,
        )

    plt.tight_layout()
    save_path = FIGURES_DIR / f"sample_predictions_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Sample predictions → {save_path}")
