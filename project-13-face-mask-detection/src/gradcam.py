"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)

Produces visual explanations of what regions a CNN focuses on
when making mask/no-mask predictions.

Reference: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import cv2
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM implementation for binary classification CNN.
    Compatible with both Keras 3 Sequential and Functional models.

    Usage:
        gcam = GradCAM(model, layer_name="conv2d_2")
        heatmap = gcam.compute(img_array)
        overlay = gcam.overlay(original_rgb, heatmap)
    """

    def __init__(self, model: Model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self._split_idx = self._find_layer_index()

    def _find_target_layer(self) -> str:
        """Auto-select last Conv2D layer."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found — pass layer_name explicitly.")

    def _find_layer_index(self) -> int:
        for i, layer in enumerate(self.model.layers):
            if layer.name == self.layer_name:
                return i
        raise ValueError(f"Layer '{self.layer_name}' not found in model.")

    def compute(self, img_array: np.ndarray, class_idx: int = 0) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            img_array: Float32 array of shape (1, H, W, 3), normalized [0,1].
            class_idx: Output neuron index (0 for binary sigmoid).

        Returns:
            2D heatmap as float32 array in [0, 1].
        """
        # Split Sequential model at the target conv layer
        front_layers = self.model.layers[:self._split_idx + 1]
        back_layers  = self.model.layers[self._split_idx + 1:]

        x = tf.constant(img_array, dtype=tf.float32)

        # Forward pass through front to get conv output
        with tf.GradientTape() as tape:
            for layer in front_layers:
                x = layer(x, training=False)
            tape.watch(x)
            conv_out = x
            # Forward through remainder
            z = conv_out
            for layer in back_layers:
                z = layer(z, training=False)
            loss = z[0, 0]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return np.zeros(conv_out.shape[1:3], dtype=np.float32)

        pooled = tf.reduce_mean(grads[0], axis=(0, 1)).numpy()
        conv_arr = conv_out[0].numpy()
        heatmap = np.sum(conv_arr * pooled[np.newaxis, np.newaxis, :], axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.astype(np.float32)

    @staticmethod
    def overlay(
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend Grad-CAM heatmap with original image.

        Args:
            original_rgb: H×W×3 uint8 RGB image.
            heatmap:      2D float heatmap.
            alpha:        Heatmap blend strength.
            colormap:     OpenCV colormap for heatmap.

        Returns:
            Blended RGB uint8 image.
        """
        h, w = original_rgb.shape[:2]
        hm_resized = cv2.resize(heatmap, (w, h))
        hm_uint8 = (hm_resized * 255).astype(np.uint8)
        hm_colored = cv2.applyColorMap(hm_uint8, colormap)
        hm_rgb = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        blended = (alpha * hm_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
        return blended


def compute_gradcam_for_batch(
    model: Model,
    images_rgb: list,
    layer_name: Optional[str] = None,
    img_size: int = 128,
) -> list:
    """
    Compute Grad-CAM for a list of RGB images.

    Args:
        model:      Trained model.
        images_rgb: List of H×W×3 uint8 RGB arrays.
        layer_name: Target layer.
        img_size:   Resize target before inference.

    Returns:
        List of (original_rgb, heatmap, label, confidence) tuples.
    """
    gcam = GradCAM(model, layer_name)
    results = []

    for img_rgb in images_rgb:
        # Preprocess
        resized = cv2.resize(img_rgb, (img_size, img_size))
        inp = resized.astype(np.float32)[np.newaxis] / 255.0

        # Predict
        conf = float(model.predict(inp, verbose=0)[0][0])
        label = "Mask" if conf >= 0.5 else "No Mask"
        display_conf = conf if conf >= 0.5 else 1 - conf

        # Grad-CAM
        heatmap = gcam.compute(inp)
        results.append((resized, heatmap, label, display_conf))

    return results
