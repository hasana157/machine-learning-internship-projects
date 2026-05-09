"""
train.py
--------
Main entry point for training both models.

Usage:
    python train.py --mode baseline          # Train baseline CNN only
    python train.py --mode transfer          # Train transfer model only
    python train.py --mode both              # Train both (default)
    python train.py --mode both --epochs 20  # Override epoch count

The script:
    1. Loads and splits CIFAR-10.
    2. Builds tf.data pipelines.
    3. Trains the selected model(s) with full callbacks.
    4. Evaluates on the test set.
    5. Saves training curves and evaluation plots.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import tensorflow as tf

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src.config import BATCH_SIZE
from src.data_loader import (
    build_baseline_dataset,
    build_tl_dataset,
    load_raw_cifar10,
    split_train_val,
)
from src.evaluate import (
    evaluate_model,
    plot_combined_curves,
    plot_sample_predictions,
    plot_training_curves,
)
from src.models import build_baseline_cnn, build_transfer_model, unfreeze_top_layers
from src.trainer import Trainer


def train_baseline(x_train, y_train, x_val, y_val, x_test, y_test, epochs=None):
    print("\n" + "="*60)
    print("  MODEL A — Baseline CNN")
    print("="*60)

    # Build datasets
    train_ds = build_baseline_dataset(x_train, y_train, training=True)
    val_ds   = build_baseline_dataset(x_val,   y_val,   training=False)
    test_ds  = build_baseline_dataset(x_test,  y_test,  training=False)

    # Build and summarise model
    model = build_baseline_cnn()
    model.summary(line_length=80)

    # Train
    trainer = Trainer("baseline", epochs=epochs)
    history = trainer.fit(model, train_ds, val_ds)

    # Evaluate
    from src.data_loader import normalise
    x_test_norm = normalise(x_test)
    results = evaluate_model(model, test_ds, x_test, y_test, model_name="Baseline CNN")

    # Plots
    plot_training_curves(history, model_name="Baseline CNN")
    plot_sample_predictions(
        model,
        x_sample=x_test_norm[:25],
        y_true=y_test[:25],
        x_display=x_test[:25],
        model_name="Baseline CNN",
    )

    return model, history, results


def train_transfer(x_train, y_train, x_val, y_val, x_test, y_test, epochs=None):
    print("\n" + "="*60)
    print("  MODEL B — Transfer Learning (MobileNetV2)")
    print("="*60)

    # Build datasets
    train_ds = build_tl_dataset(x_train, y_train, training=True)
    val_ds   = build_tl_dataset(x_val,   y_val,   training=False)
    test_ds  = build_tl_dataset(x_test,  y_test,  training=False)

    # Build model
    model   = build_transfer_model()
    trainer = Trainer("transfer", epochs=epochs)

    # Two-phase training
    history1, history2 = trainer.fit_two_phase(
        model,
        train_ds,
        val_ds,
        unfreeze_fn=unfreeze_top_layers,
        n_layers=30,
        phase2_epochs=10,
    )

    # Merge histories for plotting
    merged_history = {
        key: history1.history[key] + history2.history[key]
        for key in history1.history
    }

    # Evaluate
    from src.data_loader import normalise
    import numpy as np
    import cv2

    x_test_norm = normalise(x_test)
    # Resize test images to 96×96 for the TL model
    x_test_96 = np.stack([
        cv2.resize(img, (96, 96)) for img in x_test[:25]
    ])
    x_test_96_norm = normalise(x_test_96)

    results = evaluate_model(
        model, test_ds, x_test, y_test, model_name="Transfer MobileNetV2"
    )

    # Plots
    plot_training_curves(merged_history, model_name="Transfer MobileNetV2")
    plot_sample_predictions(
        model,
        x_sample=x_test_96_norm[:25],
        y_true=y_test[:25],
        x_display=x_test[:25],
        model_name="Transfer MobileNetV2",
    )

    # Grad-CAM
    try:
        from utils.gradcam import plot_gradcam_grid
        plot_gradcam_grid(
            model,
            x_proc=x_test_96_norm[:8],
            x_disp=x_test[:8],
            y_true=y_test[:8],
            model_name="Transfer MobileNetV2",
        )
    except Exception as e:
        print(f"[WARN] Grad-CAM skipped: {e}")

    return model, merged_history, results


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training Script")
    parser.add_argument("--mode",   default="both",
                        choices=["baseline", "transfer", "both"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count for selected model(s)")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[INFO] Loading CIFAR-10...")
    (x_train_full, y_train_full), (x_test, y_test) = load_raw_cifar10()
    (x_train, y_train), (x_val, y_val) = split_train_val(x_train_full, y_train_full)

    print(f"  Train : {x_train.shape[0]:,} samples")
    print(f"  Val   : {x_val.shape[0]:,} samples")
    print(f"  Test  : {x_test.shape[0]:,} samples")

    # ── EDA augmentation comparison ──────────────────────────────────────────
    try:
        from src.data_loader import build_augmentation_layer_baseline
        from utils.visualization import plot_augmentation_comparison
        augment_layer = build_augmentation_layer_baseline()
        plot_augmentation_comparison(x_train, augment_layer)
    except Exception as e:
        print(f"[WARN] Augmentation plot skipped: {e}")

    # ── Train selected models ─────────────────────────────────────────────────
    history_b = history_t = None

    if args.mode in ("baseline", "both"):
        model_b, history_b, _ = train_baseline(
            x_train, y_train, x_val, y_val, x_test, y_test, epochs=args.epochs
        )

    if args.mode in ("transfer", "both"):
        model_t, history_t, _ = train_transfer(
            x_train, y_train, x_val, y_val, x_test, y_test, epochs=args.epochs
        )

    # ── Comparison plot ───────────────────────────────────────────────────────
    if history_b is not None and history_t is not None:
        plot_combined_curves(history_b, history_t)

    print("\n[DONE] All done. Check reports/figures/ and models/.\n")


if __name__ == "__main__":
    main()
