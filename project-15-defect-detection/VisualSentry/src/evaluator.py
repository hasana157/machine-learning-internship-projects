"""
evaluator.py
============
Anomaly scoring, adaptive threshold computation, and evaluation metrics for
VisualSentry. Computes per-image MSE reconstruction error, derives a statistical
threshold from normal validation images, and produces overlay heatmaps.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AnomalyEvaluator:
    """Evaluates a trained ConvAutoencoder for anomaly detection.

    Computes per-image anomaly scores based on reconstruction MSE, derives an
    adaptive statistical threshold, produces pass/fail predictions, calculates
    classification metrics (precision, recall, F1, AUC-ROC), and generates
    visual heatmaps of reconstruction error.

    Attributes:
        model: Loaded Keras autoencoder model.
        threshold: Computed anomaly score threshold (mu + k*sigma).
        cfg: Configuration dictionary.
    """

    def __init__(self, model: tf.keras.Model, config_path: str = "config.yaml") -> None:
        """Initialise with a trained model.

        Args:
            model: Trained Keras autoencoder.
            config_path: Path to the YAML configuration.
        """
        self.model = model
        self.cfg = load_config(config_path)
        self.threshold: Optional[float] = None
        self._eval_cfg = self.cfg["evaluation"]
        self._model_cfg = self.cfg["model"]
        self._paths_cfg = self.cfg["paths"]

    # ── Core scoring ──────────────────────────────────────────────────────────

    def compute_anomaly_score(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute the anomaly score for a single image.

        Passes the image through the autoencoder and measures per-pixel MSE
        between the original and reconstruction. The scalar anomaly score is
        the mean over all pixels and channels.

        Args:
            image: Float32 ndarray of shape (H, W, C) in [0, 1].

        Returns:
            Tuple of (anomaly_score, error_map) where error_map is a float32
            array of shape (H, W) representing per-pixel squared error.
        """
        inp = image[np.newaxis, ...]  # (1, H, W, C)
        reconstruction = self.model.predict(inp, verbose=0)[0]  # (H, W, C)
        error_map = np.mean((image - reconstruction) ** 2, axis=-1)  # (H, W)
        score = float(np.mean(error_map))
        return score, error_map

    def compute_scores_batch(
        self, dataset: tf.data.Dataset, paths: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute anomaly scores for a full tf.data.Dataset in batches.

        Args:
            dataset: Batched dataset yielding image tensors (without labels).
            paths: Ordered list of file paths matching dataset order.

        Returns:
            Tuple of (scores, reconstructions) arrays.
        """
        all_images = []
        all_reconstructions = []

        for batch in dataset:
            recs = self.model.predict(batch, verbose=0)
            all_images.append(batch.numpy())
            all_reconstructions.append(recs)

        images = np.concatenate(all_images, axis=0)
        reconstructions = np.concatenate(all_reconstructions, axis=0)

        # Per-image mean MSE over all pixels and channels
        scores = np.mean((images - reconstructions) ** 2, axis=(1, 2, 3))
        return scores, reconstructions

    # ── Threshold computation ─────────────────────────────────────────────────

    def fit_threshold(self, normal_dataset: tf.data.Dataset) -> float:
        """Derive the anomaly decision threshold from normal validation images.

        Computes mu + k*sigma over the distribution of anomaly scores on
        known-normal images, where k is `threshold_multiplier` from config.

        Args:
            normal_dataset: tf.data.Dataset of normal-only images.

        Returns:
            The computed threshold value.
        """
        scores = []
        for batch in normal_dataset:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            recs = self.model.predict(batch, verbose=0)
            imgs = batch.numpy()
            batch_scores = np.mean((imgs - recs) ** 2, axis=(1, 2, 3))
            scores.extend(batch_scores.tolist())

        scores = np.array(scores)
        mu = float(np.mean(scores))
        sigma = float(np.std(scores))
        k = self._eval_cfg["threshold_multiplier"]
        self.threshold = mu + k * sigma

        logger.info(
            "Threshold fitted — mu=%.6f | sigma=%.6f | k=%.1f | threshold=%.6f",
            mu,
            sigma,
            k,
            self.threshold,
        )
        return self.threshold

    # ── Full evaluation run ───────────────────────────────────────────────────

    def evaluate(
        self,
        dataset: tf.data.Dataset,
        paths: List[str],
        labels: List[int],
        save_results: bool = True,
    ) -> pd.DataFrame:
        """Run full evaluation: score all images, predict, compute metrics.

        Args:
            dataset: Batched tf.data.Dataset of evaluation images.
            paths: File paths corresponding to dataset images, in order.
            labels: Ground-truth binary labels (0=normal, 1=defect).
            save_results: Whether to write results CSV to disk.

        Returns:
            DataFrame with columns: image_path, anomaly_score, label,
            predicted, confidence.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit_threshold() before evaluate().")

        scores, _ = self.compute_scores_batch(dataset, paths)

        predicted = (scores > self.threshold).astype(int)
        confidence = self._score_to_confidence(scores)

        label_names = ["normal" if l == 0 else "defect" for l in labels]
        predicted_names = ["fail" if p == 1 else "pass" for p in predicted]

        results = pd.DataFrame(
            {
                "image_path": paths,
                "anomaly_score": scores,
                "label": label_names,
                "predicted": predicted_names,
                "confidence": confidence,
                "correct": (np.array(labels) == predicted).astype(int),
            }
        )

        metrics = self.compute_metrics(labels, predicted, scores)
        logger.info(
            "Evaluation metrics — Precision=%.3f | Recall=%.3f | F1=%.3f | AUC-ROC=%.3f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auc_roc"],
        )

        if save_results:
            out_path = self._paths_cfg["evaluation_results"]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(out_path, index=False)
            logger.info("Evaluation results saved to '%s'.", out_path)

        return results

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        labels: List[int],
        predicted: np.ndarray,
        scores: np.ndarray,
    ) -> Dict[str, float]:
        """Compute classification metrics for anomaly detection.

        Args:
            labels: Ground-truth binary labels (0=normal, 1=defect).
            predicted: Predicted binary labels.
            scores: Continuous anomaly scores for AUC-ROC computation.

        Returns:
            Dictionary with keys: precision, recall, f1, auc_roc.
        """
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        labels_arr = np.array(labels)
        metrics = {
            "precision": float(precision_score(labels_arr, predicted, zero_division=0)),
            "recall": float(recall_score(labels_arr, predicted, zero_division=0)),
            "f1": float(f1_score(labels_arr, predicted, zero_division=0)),
            "auc_roc": float(roc_auc_score(labels_arr, scores)) if len(np.unique(labels_arr)) > 1 else 0.5,
        }
        return metrics

    def compute_roc_curve(
        self, labels: List[int], scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute the ROC curve and AUC for the given scores.

        Args:
            labels: Ground-truth binary labels.
            scores: Continuous anomaly scores.

        Returns:
            Tuple of (fpr, tpr, auc_value).
        """
        from sklearn.metrics import roc_curve, roc_auc_score

        labels_arr = np.array(labels)
        fpr, tpr, _ = roc_curve(labels_arr, scores)
        auc = float(roc_auc_score(labels_arr, scores))
        return fpr, tpr, auc

    def compute_pr_curve(
        self, labels: List[int], scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Precision-Recall curve.

        Args:
            labels: Ground-truth binary labels.
            scores: Continuous anomaly scores.

        Returns:
            Tuple of (precision_array, recall_array).
        """
        from sklearn.metrics import precision_recall_curve

        labels_arr = np.array(labels)
        precision, recall, _ = precision_recall_curve(labels_arr, scores)
        return precision, recall

    # ── Heatmap generation ────────────────────────────────────────────────────

    def generate_heatmap(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Generate a reconstruction error heatmap for a single image.

        Computes the per-pixel MSE error map, normalises it, and blends it
        with the original image using the configured alpha value and 'hot'
        colourmap to produce an interpretable overlay.

        Args:
            image: Float32 ndarray of shape (H, W, C) in [0, 1].

        Returns:
            Tuple of (original_uint8, overlay_uint8, anomaly_score) where the
            overlay is an RGB uint8 image suitable for display.
        """
        import matplotlib.cm as cm

        score, error_map = self.compute_anomaly_score(image)

        # Normalise error map to [0, 1]
        emin, emax = error_map.min(), error_map.max()
        if emax > emin:
            norm_map = (error_map - emin) / (emax - emin)
        else:
            norm_map = np.zeros_like(error_map)

        # Apply 'hot' colourmap → RGBA float
        cmap_fn = cm.get_cmap(self._eval_cfg.get("heatmap_colormap", "hot"))
        heatmap_rgba = cmap_fn(norm_map)[:, :, :3]  # (H, W, 3) float

        # Alpha blend with original image
        alpha = self._eval_cfg["anomaly_heatmap_alpha"]
        overlay = (1 - alpha) * image + alpha * heatmap_rgba
        overlay = np.clip(overlay, 0.0, 1.0)

        original_uint8 = (image * 255).astype(np.uint8)
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        return original_uint8, overlay_uint8, score

    def generate_heatmap_from_path(
        self, image_path: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Load an image from disk and generate its anomaly heatmap.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (original_uint8, overlay_uint8, anomaly_score).
        """
        img_size = tuple(self._model_cfg["img_size"])
        channels = self._model_cfg.get("channels", 3)
        pil = Image.open(image_path).convert("RGB").resize((img_size[1], img_size[0]))
        img = np.array(pil, dtype=np.float32) / 255.0
        if channels == 1:
            img = img[:, :, :1]
        return self.generate_heatmap(img)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _score_to_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw anomaly scores to a [0, 1] confidence value.

        Confidence represents how certain the model is about the DEFECT
        classification, scaled relative to the threshold.

        Args:
            scores: Array of raw anomaly scores.

        Returns:
            Array of confidence values in [0, 1].
        """
        if self.threshold is None or self.threshold == 0:
            return np.clip(scores / (scores.max() + 1e-8), 0.0, 1.0)
        # Sigmoid-like scaling centred at the threshold
        normalised = (scores - self.threshold) / (self.threshold + 1e-8)
        confidence = 1.0 / (1.0 + np.exp(-5.0 * normalised))
        return np.clip(confidence, 0.0, 1.0)

    @staticmethod
    def load_image_as_array(
        image_path: str,
        img_size: Tuple[int, int],
        channels: int = 3,
    ) -> np.ndarray:
        """Load an image file as a normalised float32 ndarray.

        Args:
            image_path: Path to the image file.
            img_size: Target (height, width).
            channels: Number of channels.

        Returns:
            Float32 ndarray of shape (H, W, channels) in [0, 1].
        """
        pil = Image.open(image_path).convert("RGB").resize((img_size[1], img_size[0]))
        arr = np.array(pil, dtype=np.float32) / 255.0
        if channels == 1:
            arr = arr[:, :, :1]
        return arr
