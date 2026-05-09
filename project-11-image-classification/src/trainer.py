"""
trainer.py
----------
Production-grade training pipeline with:
    - EarlyStopping          — prevents overfitting by monitoring val_loss.
    - ModelCheckpoint        — saves the best weights during training.
    - ReduceLROnPlateau      — halves learning rate when improvement stalls.
    - CSVLogger              — exports per-epoch metrics to a CSV for auditing.

Usage (programmatic):
    >>> from src.trainer import Trainer
    >>> trainer = Trainer("baseline")
    >>> history = trainer.fit(model, train_ds, val_ds)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from src.config import (
    BASELINE_MODEL_PATH,
    EPOCHS_BASELINE,
    EPOCHS_TL,
    FIGURES_DIR,
    MODELS_DIR,
    PATIENCE,
    TRANSFER_MODEL_PATH,
)


class Trainer:
    """
    Encapsulates the Keras training loop, callbacks, and model persistence.

    Args:
        mode:        "baseline" or "transfer".
        epochs:      Override default epoch count.
        verbose:     Keras verbosity (0 = silent, 1 = progress bar, 2 = one line/epoch).
    """

    MODE_CONFIG = {
        "baseline": {
            "save_path":  BASELINE_MODEL_PATH,
            "log_path":   str(MODELS_DIR / "baseline_training_log.csv"),
            "epochs":     EPOCHS_BASELINE,
        },
        "transfer": {
            "save_path":  TRANSFER_MODEL_PATH,
            "log_path":   str(MODELS_DIR / "transfer_training_log.csv"),
            "epochs":     EPOCHS_TL,
        },
    }

    def __init__(
        self,
        mode: str = "baseline",
        epochs: Optional[int] = None,
        verbose: int = 1,
    ) -> None:
        if mode not in self.MODE_CONFIG:
            raise ValueError(f"`mode` must be one of {list(self.MODE_CONFIG.keys())}")

        cfg = self.MODE_CONFIG[mode]
        self.mode      = mode
        self.save_path = cfg["save_path"]
        self.log_path  = cfg["log_path"]
        self.epochs    = epochs or cfg["epochs"]
        self.verbose   = verbose

    # ── Callback factory ─────────────────────────────────────────────────────

    def _build_callbacks(self) -> list:
        callbacks = [
            # Save the best model (by val_loss) during training
            keras.callbacks.ModelCheckpoint(
                filepath=self.save_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            # Stop when val_loss stops improving
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            # Halve LR when val_loss plateaus for 4 epochs
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1,
            ),
            # Write epoch metrics to CSV for later analysis
            keras.callbacks.CSVLogger(
                filename=self.log_path,
                append=False,
            ),
        ]
        return callbacks

    # ── Main training entry point ─────────────────────────────────────────────

    def fit(
        self,
        model: keras.Model,
        train_ds: tf.data.Dataset,
        val_ds:   tf.data.Dataset,
    ) -> keras.callbacks.History:
        """
        Run the training loop and return the Keras History object.

        Args:
            model:    Compiled Keras model.
            train_ds: Training tf.data.Dataset (batched, prefetched).
            val_ds:   Validation tf.data.Dataset (batched, prefetched).

        Returns:
            Keras History object containing per-epoch metrics.
        """
        print(f"\n{'='*60}")
        print(f"  Training [{self.mode.upper()}]  —  max {self.epochs} epochs")
        print(f"  Checkpoint → {self.save_path}")
        print(f"{'='*60}\n")

        t0 = time.time()

        history = model.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
            callbacks=self._build_callbacks(),
            verbose=self.verbose,
        )

        elapsed = time.time() - t0
        best_val_acc = max(history.history.get("val_accuracy", [0]))
        print(f"\n[DONE] Training complete in {elapsed/60:.1f} min")
        print(f"[DONE] Best val_accuracy = {best_val_acc:.4f}")
        print(f"[DONE] Model saved → {self.save_path}\n")

        return history

    # ── Two-phase transfer learning convenience method ───────────────────────

    def fit_two_phase(
        self,
        model,
        train_ds: tf.data.Dataset,
        val_ds:   tf.data.Dataset,
        unfreeze_fn,
        n_layers: int = 30,
        phase2_epochs: int = 15,
    ) -> tuple[keras.callbacks.History, keras.callbacks.History]:
        """
        Phase 1: train with frozen backbone (standard fit).
        Phase 2: unfreeze top layers, fine-tune with low LR.

        Args:
            model:         Transfer learning model.
            unfreeze_fn:   Function that unfreezes top layers (from models.py).
            n_layers:      Number of backbone layers to unfreeze in phase 2.
            phase2_epochs: Additional epochs for fine-tuning.

        Returns:
            (history_phase1, history_phase2)
        """
        print("\n── Phase 1: Training classification head (backbone frozen) ──")
        history1 = self.fit(model, train_ds, val_ds)

        print("\n── Phase 2: Fine-tuning backbone top layers ──")
        unfreeze_fn(model, n_layers)

        # Temporarily raise epoch count for phase 2
        orig_epochs  = self.epochs
        self.epochs  = phase2_epochs
        history2 = self.fit(model, train_ds, val_ds)
        self.epochs  = orig_epochs

        return history1, history2
