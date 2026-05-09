"""
train.py — Full training pipeline for the mask classifier.

Two-phase training:
  Phase 1: Frozen MobileNetV2 backbone (fast convergence)
  Phase 2: Fine-tune top layers (higher accuracy)

Bonus: Augmentation comparison (with vs. without)

Usage:
    python src/train.py --data_dir data --epochs 30
    python src/train.py --data_dir data --compare_augmentation
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Suppress TF noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
from sklearn.metrics import classification_report
from utils.data_utils import get_data_generators, IMG_SIZE, BATCH_SIZE
from utils.viz_utils import (
    plot_training_curves, plot_confusion_matrix,
    plot_class_distribution, plot_sample_grid,
    plot_augmentation_preview, plot_failure_cases,
)
from src.model import build_classifier, build_custom_cnn, unfreeze_top_layers, get_callbacks


def load_data_for_class_report(data_dir: str, img_size: int, batch_size: int):
    """Load full val set as arrays for sklearn metrics."""
    from utils.data_utils import get_data_generators
    _, val_gen, class_indices = get_data_generators(
        data_dir, img_size=img_size, batch_size=batch_size,
        augment=False
    )
    X, y = [], []
    for _ in range(len(val_gen)):
        xb, yb = next(val_gen)
        X.append(xb)
        y.append(yb)
    return np.concatenate(X), np.concatenate(y).astype(int), class_indices


def run_training(
    data_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs: int = 30,
    fine_tune_epochs: int = 10,
    model_path: str = "models/mask_classifier.keras",
    augment: bool = True,
    use_mobilenet: bool = True,
):
    """Main training function."""
    print("\n" + "=" * 60)
    print("  FACE MASK DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Loading dataset...")
    train_gen, val_gen, class_indices = get_data_generators(
        data_dir, img_size=img_size, batch_size=batch_size, augment=augment
    )
    idx_to_class = {v: k for k, v in class_indices.items()}
    print(f"      Train batches : {len(train_gen)}")
    print(f"      Val batches   : {len(val_gen)}")
    print(f"      Classes       : {class_indices}")

    # EDA plots
    from utils.data_utils import get_class_distribution, load_sample_images
    dist = get_class_distribution(data_dir)
    plot_class_distribution(dist)
    samples = load_sample_images(data_dir, n_per_class=5, img_size=img_size)
    plot_sample_grid(samples)

    # Augmentation preview
    sample_cls = list(samples.keys())[0]
    if samples[sample_cls]:
        orig = samples[sample_cls][0]
        augs = [samples[sample_cls][min(i, len(samples[sample_cls])-1)] for i in range(1, 6)]
        plot_augmentation_preview(orig, augs)

    # 2. Build model
    print("\n[2/5] Building model...")
    if use_mobilenet:
        model = build_classifier(img_size=img_size, freeze_base=True)
    else:
        model = build_custom_cnn(img_size=img_size)
    model.summary()

    # 3. Phase 1 training
    print("\n[3/5] Phase 1 — Training head (backbone frozen)...")
    callbacks = get_callbacks(model_path)
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    plot_training_curves(history1, "reports/figures/training_curves_phase1.png")

    # 4. Fine-tune (MobileNetV2 only)
    if use_mobilenet and fine_tune_epochs > 0:
        print("\n[4/5] Phase 2 — Fine-tuning top layers...")
        model = unfreeze_top_layers(model, n_layers=30)
        ft_callbacks = get_callbacks("models/mask_classifier_ft.keras")
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs,
            callbacks=ft_callbacks,
            verbose=1,
        )
        plot_training_curves(history2, "reports/figures/training_curves_finetune.png")
        model.save("models/mask_classifier_ft.keras")
        print("      Fine-tuned model saved → models/mask_classifier_ft.keras")
    else:
        print("\n[4/5] Skipping fine-tune (custom CNN).")

    # 5. Evaluation
    print("\n[5/5] Evaluating on validation set...")
    val_gen.reset()
    val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=0)
    print(f"      Val Accuracy  : {val_acc:.4f}")
    print(f"      Val AUC       : {val_auc:.4f}")
    print(f"      Val Loss      : {val_loss:.4f}")

    # Predictions
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    y_true = val_gen.classes

    class_names = [idx_to_class[0], idx_to_class[1]]
    plot_confusion_matrix(y_true, y_pred, class_names)

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # Failure cases
    _save_failure_cases(model, val_gen, y_true, y_pred, y_pred_proba, idx_to_class)

    # Save metrics
    metrics = {
        "val_accuracy": float(val_acc),
        "val_auc": float(val_auc),
        "val_loss": float(val_loss),
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best model → {model_path}")
    print("=" * 60 + "\n")
    return model, history1


def _save_failure_cases(model, val_gen, y_true, y_pred, y_pred_proba, idx_to_class):
    """Extract and save misclassified examples."""
    val_gen.reset()
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("      No failure cases found (perfect validation score)!")
        return

    wrong_imgs, wrong_true, wrong_pred, wrong_conf = [], [], [], []
    batch_size = val_gen.batch_size
    accumulated = 0

    for xb, _ in val_gen:
        for j in range(len(xb)):
            global_idx = accumulated + j
            if global_idx in wrong_idx:
                wrong_imgs.append((xb[j] * 255).astype("uint8"))
                wrong_true.append(idx_to_class[int(y_true[global_idx])])
                wrong_pred.append(idx_to_class[int(y_pred[global_idx])])
                conf = float(y_pred_proba[global_idx][0])
                wrong_conf.append(conf if conf >= 0.5 else 1 - conf)
        accumulated += len(xb)
        if accumulated >= len(y_true):
            break

    plot_failure_cases(wrong_imgs, wrong_true, wrong_pred, wrong_conf)
    print(f"      Failure cases saved → reports/figures/failure_cases.png")


def compare_augmentation(data_dir: str, epochs: int = 15):
    """Train two models and compare performance with/without augmentation."""
    print("\n" + "=" * 60)
    print("  AUGMENTATION COMPARISON")
    print("=" * 60)

    results = {}
    for aug in [True, False]:
        tag = "with_aug" if aug else "no_aug"
        print(f"\n── Training {tag} ──")
        model = build_custom_cnn(IMG_SIZE)
        train_gen, val_gen, _ = get_data_generators(
            data_dir, augment=aug, batch_size=32
        )
        history = model.fit(train_gen, validation_data=val_gen,
                            epochs=epochs, verbose=0)
        val_acc = max(history.history["val_accuracy"])
        results[tag] = {
            "val_accuracy": val_acc,
            "history": {
                "accuracy": history.history["accuracy"],
                "val_accuracy": history.history["val_accuracy"],
            }
        }
        print(f"  Best val_accuracy ({tag}): {val_acc:.4f}")

    # Plot comparison
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, r in results.items():
        ax.plot(r["history"]["val_accuracy"], label=f"Val Acc ({tag})", linewidth=2)
    ax.set_title("Augmentation Comparison — Validation Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig("reports/figures/augmentation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[VIZ] Augmentation comparison → reports/figures/augmentation_comparison.png")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask Classifier")
    parser.add_argument("--data_dir", default="data", help="Path to data dir (mask/ no_mask/)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--custom_cnn", action="store_true", help="Use custom CNN instead of MobileNetV2")
    parser.add_argument("--compare_augmentation", action="store_true")
    args = parser.parse_args()

    if args.compare_augmentation:
        compare_augmentation(args.data_dir, epochs=args.epochs)
    else:
        run_training(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            augment=not args.no_augmentation,
            use_mobilenet=not args.custom_cnn,
        )
