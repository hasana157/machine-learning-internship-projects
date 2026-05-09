"""
data_loader.py
--------------
Handles all data loading, preprocessing, and augmentation for CIFAR-10.

Why CIFAR-10?
    - 60 000 labelled 32×32 colour images across 10 classes (6 000/class).
    - Small enough to train on a laptop/Colab GPU, large enough to surface
      real generalisation challenges.
    - Widely used benchmark → results are directly comparable to literature.

Limitations of CIFAR-10:
    - Very low resolution (32×32) — fine-grained textures are lost.
    - Only 10 coarse classes — unsuitable for fine-grained recognition tasks.
    - Significant class overlap (cat/dog, automobile/truck) hurts accuracy.
    - Data was collected in 2009; distribution may not reflect modern imagery.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

from src.config import (
    BATCH_SIZE,
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGE_SIZE_TL,
    NUM_CLASSES,
    SEED,
    VALIDATION_SPLIT,
)


# ── Normalisation ────────────────────────────────────────────────────────────

def normalise(images: np.ndarray) -> np.ndarray:
    """
    Channel-wise standardisation using CIFAR-10 training-set statistics.
    Subtracting the mean centres the data; dividing by the std scales it.
    This helps gradient descent converge faster and more stably.
    """
    images = images.astype(np.float32) / 255.0
    mean = np.array(CIFAR10_MEAN, dtype=np.float32)
    std  = np.array(CIFAR10_STD,  dtype=np.float32)
    return (images - mean) / std


# ── Raw data loading ─────────────────────────────────────────────────────────

def load_raw_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 and return (x_train, y_train), (x_test, y_test)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    return (x_train, y_train), (x_test, y_test)


# ── Augmentation layer (baseline CNN — native 32×32 input) ──────────────────

def build_augmentation_layer_baseline() -> keras.Sequential:
    """
    Light augmentation applied *only during training* via Keras preprocessing layers.
    Running augmentation on-the-fly (vs. offline) keeps disk usage minimal and
    exposes the model to a new random variant every epoch.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        keras.layers.RandomRotation(factor=0.1),
        keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
    ], name="augmentation_baseline")


def build_augmentation_layer_tl() -> keras.Sequential:
    """
    Stronger augmentation for the transfer learning model (96×96 inputs).
    MobileNetV2 was pretrained on ImageNet imagery which is richer and more
    varied than CIFAR-10, so stronger regularisation reduces overfitting.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
        keras.layers.RandomRotation(factor=0.15),
        keras.layers.RandomZoom(height_factor=(-0.15, 0.15)),
        keras.layers.RandomContrast(factor=0.2),
    ], name="augmentation_tl")


# ── tf.data pipeline builders ────────────────────────────────────────────────

def build_baseline_dataset(
    x: np.ndarray,
    y: np.ndarray,
    training: bool = True,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline for the baseline CNN (native 32×32 images).

    Args:
        x:          Image array of shape (N, 32, 32, 3), uint8.
        y:          Integer label array of shape (N,).
        training:   If True, apply augmentation and shuffle.
        batch_size: Mini-batch size.

    Returns:
        A batched, prefetched tf.data.Dataset.
    """
    x_norm = normalise(x)
    ds = tf.data.Dataset.from_tensor_slices((x_norm, y))

    if training:
        ds = ds.shuffle(buffer_size=len(x), seed=SEED)

    ds = ds.batch(batch_size)

    if training:
        augment = build_augmentation_layer_baseline()
        ds = ds.map(
            lambda imgs, labels: (augment(imgs, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


def build_tl_dataset(
    x: np.ndarray,
    y: np.ndarray,
    training: bool = True,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline for the transfer learning model (upsampled to 96×96).

    MobileNetV2 expects inputs in [-1, 1] which we handle via its own
    `preprocess_input` function inside the model graph, so here we only
    resize and normalise to [0, 1].
    """
    x_norm = normalise(x)
    ds = tf.data.Dataset.from_tensor_slices((x_norm, y))

    if training:
        ds = ds.shuffle(buffer_size=len(x), seed=SEED)

    # Resize on-the-fly to keep memory footprint low
    target_h, target_w = IMAGE_SIZE_TL
    ds = ds.map(
        lambda img, label: (
            tf.image.resize(img, [target_h, target_w]),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)

    if training:
        augment = build_augmentation_layer_tl()
        ds = ds.map(
            lambda imgs, labels: (augment(imgs, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


# ── Train / val split helper ─────────────────────────────────────────────────

def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Randomly split training data into train / validation sets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    val_size = int(len(x) * val_split)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_val_idx := val_idx])


# Fix the function above (clean version):
def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Randomly split training data into train / validation sets."""
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    n_val   = int(len(x) * val_split)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])
