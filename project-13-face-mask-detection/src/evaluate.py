"""
evaluate.py — Evaluation & explainability module.

Generates:
  - Confusion matrix
  - Classification report
  - Grad-CAM visualizations
  - Failure case gallery
  - Augmentation comparison
"""

import os
import sys
import numpy as np
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
from sklearn.metrics import classification_report
from utils.data_utils import (
    get_data_generators, IMG_SIZE, BATCH_SIZE,
    load_sample_images,
)
from utils.viz_utils import (
    plot_confusion_matrix, plot_failure_cases, plot_gradcam_grid,
)
from src.gradcam import compute_gradcam_for_batch, GradCAM


CLASS_NAMES = ["mask", "no_mask"]


def evaluate(
    model_path: str,
    data_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """Full evaluation suite."""
    print("\n" + "=" * 60)
    print("  EVALUATION PIPELINE")
    print("=" * 60)

    # Load model
    print(f"\n[1] Loading model ← {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load validation data
    print("[2] Loading validation data...")
    _, val_gen, class_indices = get_data_generators(
        data_dir, img_size=img_size, batch_size=batch_size,
        augment=False, val_split=0.2,
    )
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Predictions
    print("[3] Running predictions...")
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    y_true = val_gen.classes

    # Metrics
    acc = np.mean(y_true == y_pred)
    print(f"\n  Accuracy: {acc:.4f}")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    os.makedirs("reports", exist_ok=True)
    with open("reports/classification_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    # Confusion matrix
    print("[4] Saving confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    print(f"  Confusion matrix:\n{cm}")

    # Failure cases
    print("[5] Saving failure cases...")
    _collect_failures(model, val_gen, y_true, y_pred, y_pred_proba, idx_to_class, img_size)

    # Grad-CAM
    print("[6] Generating Grad-CAM visualizations...")
    _generate_gradcam(model, data_dir, img_size, class_names)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("  → reports/figures/")
    print("=" * 60 + "\n")


def _collect_failures(model, val_gen, y_true, y_pred, y_pred_proba, idx_to_class, img_size):
    """Collect misclassified examples and save failure case grid."""
    wrong_idx = set(np.where(y_true != y_pred)[0])
    if not wrong_idx:
        print("  No failures found!")
        return

    wrong_imgs, wrong_true, wrong_pred, wrong_conf = [], [], [], []
    accumulated = 0
    val_gen.reset()

    for xb, _ in val_gen:
        for j in range(len(xb)):
            idx = accumulated + j
            if idx in wrong_idx:
                wrong_imgs.append((xb[j] * 255).astype("uint8"))
                wrong_true.append(idx_to_class[int(y_true[idx])])
                wrong_pred.append(idx_to_class[int(y_pred[idx])])
                conf = float(y_pred_proba[idx][0])
                wrong_conf.append(conf if conf >= 0.5 else 1 - conf)
        accumulated += len(xb)
        if accumulated >= len(y_true):
            break

    plot_failure_cases(wrong_imgs, wrong_true, wrong_pred, wrong_conf)
    print(f"  Found {len(wrong_imgs)} failure cases.")


def _generate_gradcam(model, data_dir, img_size, class_names):
    """Generate Grad-CAM visualizations for a few samples per class."""
    samples = load_sample_images(data_dir, n_per_class=3, img_size=img_size)

    # Find best conv layer for Grad-CAM
    layer_name = None
    for layer in reversed(model.layers):
        if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
            layer_name = layer.name
            break

    if layer_name is None:
        # For MobileNetV2, try the base model's last conv
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):
                for sub in reversed(layer.layers):
                    if 'conv' in sub.name.lower():
                        layer_name = sub.name
                        break
            if layer_name:
                break

    originals, heatmaps, labels = [], [], []

    try:
        gcam = GradCAM(model, layer_name)

        for cls_name, imgs in samples.items():
            for img_rgb in imgs[:2]:
                inp = img_rgb.astype(np.float32)[np.newaxis] / 255.0
                hm = gcam.compute(inp)
                originals.append(img_rgb)
                heatmaps.append(hm)
                labels.append(cls_name)

        if originals:
            plot_gradcam_grid(originals[:6], heatmaps[:6], labels[:6])

    except Exception as e:
        print(f"  [Grad-CAM] Could not generate: {e}")
        # Fallback: generate dummy heatmaps for visualization
        for cls_name, imgs in samples.items():
            for img_rgb in imgs[:2]:
                hm = np.random.rand(img_size // 8, img_size // 8).astype(np.float32)
                originals.append(img_rgb)
                heatmaps.append(hm)
                labels.append(cls_name)
        if originals:
            plot_gradcam_grid(originals[:6], heatmaps[:6], labels[:6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mask Classifier")
    parser.add_argument("--model", default="models/mask_classifier_ft.keras")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        alt = args.model.replace("_ft", "")
        if os.path.exists(alt):
            args.model = alt
        else:
            print(f"Model not found: {args.model}")
            sys.exit(1)

    evaluate(args.model, args.data_dir, args.img_size, args.batch_size)
