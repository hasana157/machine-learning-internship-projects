"""
trainer.py
==========
Training orchestration for VisualSentry's ConvAutoencoder.

Provides an end-to-end Trainer class that wires together Keras callbacks
(EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger), runs the
training loop, and persists artefacts (model weights, loss curve, training log).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import yaml

from src.model import ConvAutoencoder
from src.data_loader import build_train_dataset, generate_demo_data, load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Trainer:
    """End-to-end training orchestrator for the ConvAutoencoder.

    Attributes:
        cfg: Full configuration dictionary loaded from config.yaml.
        model: ConvAutoencoder instance.
        history: Keras History object returned after fitting.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialise Trainer with configuration.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.cfg = load_config(config_path)
        self.config_path = config_path
        self.model: Optional[ConvAutoencoder] = None
        self.history = None

        # Ensure output directories exist
        Path(self.cfg["paths"]["reports"]).mkdir(parents=True, exist_ok=True)
        Path(self.cfg["paths"]["figures"]).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.cfg["paths"]["model_save"])).mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_model(self) -> ConvAutoencoder:
        """Instantiate and compile the ConvAutoencoder.

        Returns:
            Compiled ConvAutoencoder ready for training.
        """
        self.model = ConvAutoencoder.from_config_file(self.config_path)
        self.model.build_graph()

        lr = self.cfg["training"]["learning_rate"]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"],
        )
        logger.info("Model compiled — latent_dim=%d", self.cfg["model"]["latent_dim"])
        return self.model

    def train(
        self,
        progress_callback=None,
    ) -> tf.keras.callbacks.History:
        """Run the full training pipeline.

        Builds datasets, constructs callbacks, fits the model, saves artefacts.

        Args:
            progress_callback: Optional callable(epoch, logs) invoked after each epoch
                               (e.g. to update a Streamlit progress bar).

        Returns:
            Keras History object containing per-epoch metrics.
        """
        if self.model is None:
            self.build_model()

        train_cfg = self.cfg["training"]
        data_cfg = self.cfg["data"]
        paths_cfg = self.cfg["paths"]
        model_cfg = self.cfg["model"]

        img_size = tuple(model_cfg["img_size"])
        channels = model_cfg.get("channels", 3)

        # Verify data exists; auto-generate demo data if not
        normal_dir = paths_cfg["normal_data"]
        if not list(Path(normal_dir).glob("*.png")) + list(Path(normal_dir).glob("*.jpg")):
            logger.warning("No images found in '%s'. Generating demo data …", normal_dir)
            generate_demo_data(self.config_path)

        train_ds, val_ds = build_train_dataset(
            normal_dir=normal_dir,
            img_size=img_size,
            batch_size=train_cfg["batch_size"],
            validation_split=train_cfg["validation_split"],
            augment_cfg=data_cfg["augmentation"],
            channels=channels,
            seed=train_cfg.get("seed", 42),
        )

        callbacks = self._build_callbacks(paths_cfg, train_cfg, progress_callback)

        logger.info("Starting training for up to %d epochs …", train_cfg["epochs"])
        self.history = self.model.fit(
            train_ds,
            epochs=train_cfg["epochs"],
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )

        self._save_loss_curve(paths_cfg["loss_curve"])
        logger.info("Training complete. Model saved to '%s'.", paths_cfg["model_save"])
        return self.history

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_callbacks(
        self,
        paths_cfg: dict,
        train_cfg: dict,
        progress_callback=None,
    ) -> list:
        """Construct the list of Keras callbacks for training.

        Args:
            paths_cfg: Paths sub-dict from config.
            train_cfg: Training sub-dict from config.
            progress_callback: Optional external callback callable.

        Returns:
            List of tf.keras.callbacks.Callback instances.
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=train_cfg["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
            filepath=paths_cfg["model_save"],
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=train_cfg["lr_reduce_factor"],
                patience=train_cfg["lr_reduce_patience"],
                min_lr=train_cfg["min_lr"],
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                paths_cfg["training_log"],
                separator=",",
                append=False,
            ),
        ]

        if progress_callback is not None:
            callbacks.append(_ExternalProgressCallback(progress_callback))

        return callbacks

    def _save_loss_curve(self, output_path: str) -> None:
        """Plot and save the training/validation loss curve.

        Args:
            output_path: File path for the output PNG image.
        """
        if self.history is None:
            logger.warning("No training history found — skipping loss curve.")
            return

        hist = self.history.history
        epochs = range(1, len(hist["loss"]) + 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#0D1B2A")
        fig.patch.set_facecolor("#0D1B2A")

        ax.plot(epochs, hist["loss"], color="#00C9A7", linewidth=2.0, label="Train Loss")
        if "val_loss" in hist:
            ax.plot(epochs, hist["val_loss"], color="#FF4B4B", linewidth=2.0, linestyle="--", label="Val Loss")

        ax.set_title("VisualSentry — Training Loss", color="white", fontsize=14, pad=12)
        ax.set_xlabel("Epoch", color="#AAAAAA")
        ax.set_ylabel("MSE Loss", color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.legend(facecolor="#1A2B3C", labelcolor="white", framealpha=0.9)
        ax.grid(True, color="#1A2B3C", linewidth=0.6)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Loss curve saved to '%s'.", output_path)

    def get_training_log(self) -> Optional[pd.DataFrame]:
        """Load and return the CSV training log as a DataFrame.

        Returns:
            DataFrame with columns [epoch, loss, val_loss, mae, val_mae],
            or None if the log has not yet been created.
        """
        log_path = self.cfg["paths"]["training_log"]
        if not Path(log_path).exists():
            return None
        return pd.read_csv(log_path)


# ── Helper callback ────────────────────────────────────────────────────────────

class _ExternalProgressCallback(tf.keras.callbacks.Callback):
    """Thin Keras callback that forwards epoch-end metrics to an external callable.

    Args:
        fn: Callable accepting (epoch: int, logs: dict).
    """

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Invoke the external callable with epoch index and metric logs.

        Args:
            epoch: Zero-based epoch index.
            logs: Dictionary of metric name → value for this epoch.
        """
        self._fn(epoch, logs or {})
