"""
gradcam.py
----------
Gradient-weighted Class Activation Mapping (Grad-CAM).

Grad-CAM highlights the regions of an input image that were most
influential in the model's prediction.  For each target class c:

    1. Compute the gradient of the class score y^c w.r.t. the feature
       maps A^k of the last convolutional layer.
    2. Global-average-pool the gradients to get importance weights α^c_k.
    3. Compute a weighted combination of feature maps and ReLU it:
           L^c = ReLU(Σ_k α^c_k · A^k)
    4. Upsample L^c to the input resolution and overlay on the image.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
           Networks via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from src.config import CIFAR10_CLASSES, FIGURES_DIR


def _find_last_conv_layer(model: keras.Model) -> str:
    """Return the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def compute_gradcam(
    model: keras.Model,
    img_array: np.ndarray,          # Preprocessed image (1, H, W, 3)
    class_idx: int | None = None,   # None → use top prediction
    conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Compute the Grad-CAM heatmap for a single image.

    Args:
        model:           Keras model.
        img_array:       Preprocessed image tensor, shape (1, H, W, 3).
        class_idx:       Target class index (None = predicted class).
        conv_layer_name: Which conv layer to hook (None = auto-detect last).

    Returns:
        Heatmap as float32 numpy array in [0, 1], shape (H, W).
    """
    if conv_layer_name is None:
        conv_layer_name = _find_last_conv_layer(model)

    # Build a sub-model that outputs both conv activations and final predictions
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs      = tf.cast(img_array, tf.float32)
        conv_out, preds = grad_model(inputs)
        if class_idx is None:
            class_idx = tf.argmax(preds[0]).numpy()
        loss = preds[:, class_idx]

    # Gradients of the target class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_out)                        # (1, h, w, k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (k,)

    # Weight feature maps and collapse along channel dimension
    conv_out = conv_out[0]                                       # (h, w, k)
    heatmap  = conv_out @ pooled_grads[..., tf.newaxis]         # (h, w, 1)
    heatmap  = tf.squeeze(heatmap)                              # (h, w)

    # ReLU (keep only positive contributions) + normalise
    heatmap  = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap.astype(np.float32)


def overlay_gradcam(
    img_uint8: np.ndarray,          # Raw display image (H, W, 3) uint8
    heatmap:   np.ndarray,          # Output of compute_gradcam()
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Superimpose a Grad-CAM heatmap onto the original image.

    Returns:
        Overlaid image as uint8 numpy array (H, W, 3).
    """
    h, w = img_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colormap        = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap        = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * colormap + (1 - alpha) * img_uint8).astype(np.uint8)
    return overlay


def plot_gradcam_grid(
    model: keras.Model,
    x_proc: np.ndarray,        # Preprocessed images (N, H, W, 3)
    x_disp: np.ndarray,        # Raw uint8 images (N, H, W, 3)
    y_true: np.ndarray,
    model_name: str = "model",
    n_samples: int = 8,
    conv_layer_name: str | None = None,
) -> None:
    """
    Save a grid of [original | Grad-CAM overlay] pairs for n_samples images.
    """
    fig, axes = plt.subplots(n_samples, 2, figsize=(7, n_samples * 3))
    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=14)

    preds = np.argmax(model.predict(x_proc[:n_samples], verbose=0), axis=1)

    for i in range(n_samples):
        img_proc   = x_proc[i:i+1]
        img_disp   = x_disp[i]
        pred_class = preds[i]

        heatmap = compute_gradcam(model, img_proc, class_idx=pred_class,
                                  conv_layer_name=conv_layer_name)

        # Resize display image to match processed input spatial dims
        h_proc = img_proc.shape[1]
        img_disp_resized = cv2.resize(img_disp, (h_proc, h_proc))
        overlay = overlay_gradcam(img_disp_resized, heatmap)

        axes[i, 0].imshow(img_disp_resized)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"True: {CIFAR10_CLASSES[y_true[i]]}", fontsize=9
        )

        axes[i, 1].imshow(overlay)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(
            f"Pred: {CIFAR10_CLASSES[pred_class]}", fontsize=9
        )

    plt.tight_layout()
    save_path = FIGURES_DIR / f"gradcam_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Grad-CAM grid → {save_path}")
