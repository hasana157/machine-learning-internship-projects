"""
visualization.py
----------------
Reusable plotting utilities for EDA, augmentation comparison,
and per-class analysis.  All figures are saved to reports/figures/.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from src.config import CIFAR10_CLASSES, FIGURES_DIR


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_class_distribution(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Bar chart of sample counts per class for both splits."""
    train_counts = np.bincount(y_train, minlength=10)
    test_counts  = np.bincount(y_test,  minlength=10)
    x = np.arange(10)

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    bars_train = ax.bar(x - width/2, train_counts, width, label="Train", alpha=0.85, color="#4C72B0")
    bars_test  = ax.bar(x + width/2, test_counts,  width, label="Test",  alpha=0.85, color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Number of Samples")
    ax.set_title("CIFAR-10 Class Distribution — Train vs Test", fontsize=14)
    ax.legend()
    ax.yaxis.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate bars with counts
    for bar in bars_train:
        ax.annotate(
            f"{int(bar.get_height()):,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    _save(fig, "class_distribution.png")


def plot_sample_images(
    x: np.ndarray,
    y: np.ndarray,
    n_per_class: int = 5,
) -> None:
    """Grid of sample images — one row per class, n_per_class columns."""
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 2, 22))
    fig.suptitle("CIFAR-10 Sample Images", fontsize=16, y=1.01)

    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        class_images = x[y == cls_idx]
        # Random sample
        chosen = class_images[np.random.choice(len(class_images), n_per_class, replace=False)]
        for col, img in enumerate(chosen):
            ax = axes[cls_idx, col]
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=11, rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    _save(fig, "sample_images.png")


def plot_pixel_intensity_distribution(x: np.ndarray) -> None:
    """
    Per-channel pixel intensity histograms over the entire dataset.
    Helps confirm normalisation is appropriate and that colour channels
    have distinct statistics.
    """
    channels = ["Red", "Green", "Blue"]
    colours  = ["#e74c3c", "#2ecc71", "#3498db"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CIFAR-10 Pixel Intensity Distribution (per channel)", fontsize=13)

    x_flat = x.reshape(-1, 3)          # (N*32*32, 3)
    for i, (ch, col) in enumerate(zip(channels, colours)):
        axes[i].hist(x_flat[:, i], bins=64, color=col, alpha=0.75, edgecolor="none")
        axes[i].set_title(ch)
        axes[i].set_xlabel("Pixel Value (0–255)")
        axes[i].set_ylabel("Frequency")
        axes[i].set_xlim(0, 255)
        axes[i].yaxis.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, "pixel_intensity_distribution.png")


# ── Augmentation comparison ───────────────────────────────────────────────────

def plot_augmentation_comparison(
    x: np.ndarray,
    augment_fn,
    n: int = 6,
) -> None:
    """
    Show original vs augmented pairs for n randomly chosen images.
    Demonstrates that augmentation preserves label-relevant content
    while introducing meaningful variation.
    """
    import tensorflow as tf

    idx = np.random.choice(len(x), n, replace=False)
    originals = x[idx]

    # Normalise to [0, 1] before augmenting (augment layer expects floats)
    norm = originals.astype(np.float32) / 255.0
    batch = tf.expand_dims(norm[0], 0)
    augmented = [
        np.clip(augment_fn(tf.expand_dims(norm[i], 0), training=True).numpy()[0], 0, 1)
        for i in range(n)
    ]

    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
    fig.suptitle("Data Augmentation — Original vs Augmented", fontsize=13)

    for i in range(n):
        axes[0, i].imshow(originals[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)

        axes[1, i].imshow(augmented[i])
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Augmented", fontsize=10)

    plt.tight_layout()
    _save(fig, "augmentation_comparison.png")


# ── Internal helper ───────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")
