"""
viz_utils.py — All visualization helpers: training curves, confusion matrix,
bounding box overlay, Grad-CAM heatmaps, failure cases.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import confusion_matrix, classification_report


# ─── Palette ──────────────────────────────────────────────────────────────────
COLOR_MASK    = (0, 200, 80)      # green  (BGR for OpenCV)
COLOR_NO_MASK = (0, 60, 220)      # red
COLOR_BOX     = (255, 165, 0)     # orange fallback


# ─── Training Curves ──────────────────────────────────────────────────────────
def plot_training_curves(history, save_path: str = "reports/figures/training_curves.png"):
    """Plot accuracy and loss over epochs for train and validation."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold", y=1.02)

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Acc", linewidth=2, color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc", linewidth=2,
                 color="#FF5722", linestyle="--")
    axes[0].set_title("Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2, color="#4CAF50")
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2,
                 color="#FF9800", linestyle="--")
    axes[1].set_title("Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Training curves saved → {save_path}")


# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = "reports/figures/confusion_matrix.png",
):
    """Render and save a styled confusion matrix heatmap."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Normalized"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(f"Confusion Matrix ({title})", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Confusion matrix saved → {save_path}")
    return cm


# ─── Class Distribution ───────────────────────────────────────────────────────
def plot_class_distribution(
    dist: Dict[str, int],
    save_path: str = "reports/figures/class_distribution.png",
):
    """Bar chart of class distribution."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(list(dist.keys()), list(dist.values()), color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, dist.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Dataset Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Class")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Class distribution saved → {save_path}")


# ─── Sample Grid ──────────────────────────────────────────────────────────────
def plot_sample_grid(
    samples: Dict[str, List[np.ndarray]],
    save_path: str = "reports/figures/sample_images.png",
):
    """Grid of sample images per class."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    classes = list(samples.keys())
    n_cols = max((len(v) for v in samples.values()), default=0)
    n_rows = len(classes)

    # If there are no sample images (n_cols == 0), skip plotting gracefully.
    if n_cols == 0 or n_rows == 0:
        print(f"[VIZ] No sample images found (samples keys={classes}); skipping sample grid: {save_path}")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold")

    # Normalize axes to a 2D-indexable structure
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, cls in enumerate(classes):
        imgs = samples[cls]
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(imgs):
                ax.imshow(imgs[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(cls, fontsize=11, fontweight="bold")
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Sample grid saved → {save_path}")


# ─── Bounding Box Overlay ─────────────────────────────────────────────────────
def draw_prediction(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    face_idx: int = 0,
) -> np.ndarray:
    """
    Draw a detection box with label and confidence score on image (BGR).

    Args:
        image: BGR image array.
        box: (x1, y1, x2, y2) bounding box coordinates.
        label: "Mask" or "No Mask".
        confidence: float in [0, 1].
        face_idx: Index for multi-face display.
    """
    x1, y1, x2, y2 = box
    color = COLOR_MASK if label.lower() == "mask" else COLOR_NO_MASK

    # Box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = f"{label}: {confidence:.1%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(y1 - 10, th + 10)
    cv2.rectangle(image, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 4), color, -1)

    # Text
    cv2.putText(image, text, (x1 + 4, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def annotate_image(
    image_bgr: np.ndarray,
    detections: List[Dict],
) -> np.ndarray:
    """
    Apply all detections to a copy of the image.

    detections: list of dicts with keys: box, label, confidence
    """
    out = image_bgr.copy()
    for i, det in enumerate(detections):
        out = draw_prediction(out, det["box"], det["label"], det["confidence"], i)
    return out


# ─── Augmentation Preview ─────────────────────────────────────────────────────
def plot_augmentation_preview(
    original: np.ndarray,
    augmented_list: List[np.ndarray],
    save_path: str = "reports/figures/augmentation_preview.png",
):
    """Show original vs multiple augmented versions."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n = len(augmented_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3))
    fig.suptitle("Data Augmentation Samples", fontsize=13, fontweight="bold")

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for i, aug in enumerate(augmented_list):
        axes[i + 1].imshow(aug)
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=10)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Augmentation preview saved → {save_path}")


# ─── Failure Cases ────────────────────────────────────────────────────────────
def plot_failure_cases(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: str = "reports/figures/failure_cases.png",
    max_cases: int = 8,
):
    """Display a grid of misclassified images."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n = min(max_cases, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    fig.suptitle("Failure Cases (Misclassified)", fontsize=14, fontweight="bold", color="red")

    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(images[i])
            ax.set_title(
                f"True: {true_labels[i]}\nPred: {pred_labels[i]} ({confidences[i]:.1%})",
                fontsize=9, color="red"
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Failure cases saved → {save_path}")


# ─── Grad-CAM Overlay ─────────────────────────────────────────────────────────
def overlay_gradcam(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend Grad-CAM heatmap onto original image."""
    heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
    return blended


def plot_gradcam_grid(
    originals: List[np.ndarray],
    heatmaps: List[np.ndarray],
    labels: List[str],
    save_path: str = "reports/figures/gradcam.png",
):
    """Save a grid of original + Grad-CAM overlay pairs."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n = len(originals)
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    fig.suptitle("Grad-CAM Explainability", fontsize=14, fontweight="bold")

    if n == 1:
        axes = [axes]

    for i, (orig, hm, lbl) in enumerate(zip(originals, heatmaps, labels)):
        overlay = overlay_gradcam(orig, hm)
        axes[i][0].imshow(orig)
        axes[i][0].set_title(f"Original ({lbl})", fontsize=10)
        axes[i][0].axis("off")
        axes[i][1].imshow(overlay)
        axes[i][1].set_title("Grad-CAM", fontsize=10)
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Grad-CAM grid saved → {save_path}")
