"""
data_loader.py
==============
Data ingestion, augmentation, and synthetic dataset generation pipeline for
VisualSentry. Provides tf.data.Dataset pipelines for training and evaluation,
plus a standalone demo-data generator that requires no external downloads.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration parameters.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _parse_image(
    file_path: str,
    img_size: Tuple[int, int],
    channels: int = 3,
) -> tf.Tensor:
    """Read a single image file, decode, resize, and normalise to [0, 1].

    Args:
        file_path: Path string (or tensor) to the image file.
        img_size: Target (height, width) for resizing.
        channels: Number of colour channels (3 for RGB).

    Returns:
        Float32 tensor of shape (img_size[0], img_size[1], channels).
    """
    raw = tf.io.read_file(file_path)
    img = tf.image.decode_image(raw, channels=channels, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([img_size[0], img_size[1], channels])
    return img


def _augment(image: tf.Tensor, cfg: dict) -> tf.Tensor:
    """Apply stochastic augmentations to a training image.

    Args:
        image: Float32 image tensor in [0, 1].
        cfg: Augmentation sub-dict from config (data.augmentation).

    Returns:
        Augmented image tensor.
    """
    if cfg.get("horizontal_flip", True):
        image = tf.image.random_flip_left_right(image)
    delta = cfg.get("brightness_delta", 0.1)
    image = tf.image.random_brightness(image, max_delta=delta)
    lower = cfg.get("contrast_lower", 0.9)
    upper = cfg.get("contrast_upper", 1.1)
    image = tf.image.random_contrast(image, lower=lower, upper=upper)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def build_train_dataset(
    normal_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    validation_split: float,
    augment_cfg: dict,
    channels: int = 3,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build train and validation tf.data.Datasets from normal-only images.

    The dataset uses only normal (defect-free) images because the autoencoder
    is trained in an unsupervised fashion to reconstruct normal patterns only.

    Args:
        normal_dir: Directory containing normal training images.
        img_size: Target image size (height, width).
        batch_size: Mini-batch size.
        validation_split: Fraction of data held out for validation.
        augment_cfg: Augmentation configuration dictionary.
        channels: Number of image channels.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    paths = sorted(
        [str(p) for p in Path(normal_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not paths:
        raise FileNotFoundError(f"No images found in '{normal_dir}'. Run 'make data' first.")

    np.random.seed(seed)
    np.random.shuffle(paths)
    split_idx = int(len(paths) * (1 - validation_split))
    train_paths, val_paths = paths[:split_idx], paths[split_idx:]

    logger.info("Dataset split — train: %d | val: %d", len(train_paths), len(val_paths))

    def _load_and_augment(fp: str) -> Tuple[tf.Tensor, tf.Tensor]:
        img = _parse_image(fp, img_size, channels)
        img = _augment(img, augment_cfg)
        return img, img  # autoencoder: input == target

    def _load_only(fp: str) -> Tuple[tf.Tensor, tf.Tensor]:
        img = _parse_image(fp, img_size, channels)
        return img, img

    autotune = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_paths)
        .map(_load_and_augment, num_parallel_calls=autotune)
        .shuffle(buffer_size=len(train_paths), seed=seed)
        .batch(batch_size)
        .prefetch(autotune)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_paths)
        .map(_load_only, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )

    _log_dataset_stats(train_paths + val_paths, img_size, channels)
    return train_ds, val_ds


def build_eval_dataset(
    normal_dir: str,
    defect_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    channels: int = 3,
) -> Tuple[tf.data.Dataset, list, list]:
    """Build an evaluation dataset containing both normal and defective images.

    Args:
        normal_dir: Directory with normal (defect-free) evaluation images.
        defect_dir: Directory with defective images.
        img_size: Target image size (height, width).
        batch_size: Mini-batch size.
        channels: Number of image channels.

    Returns:
        Tuple of (dataset, file_paths, labels) where labels are 0=normal, 1=defect.
    """
    normal_paths = sorted(
        [str(p) for p in Path(normal_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    defect_paths = sorted(
        [str(p) for p in Path(defect_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    all_paths = normal_paths + defect_paths
    labels = [0] * len(normal_paths) + [1] * len(defect_paths)

    logger.info("Eval set — normal: %d | defect: %d", len(normal_paths), len(defect_paths))

    autotune = tf.data.AUTOTUNE

    ds = (
        tf.data.Dataset.from_tensor_slices(all_paths)
        .map(lambda fp: _parse_image(fp, img_size, channels), num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )

    return ds, all_paths, labels


def _log_dataset_stats(paths: list, img_size: Tuple[int, int], channels: int) -> None:
    """Log basic dataset statistics by sampling a subset of images.

    Args:
        paths: List of image file paths.
        img_size: Expected image size.
        channels: Number of channels.
    """
    sample_size = min(50, len(paths))
    sample = np.random.choice(paths, sample_size, replace=False)
    pixel_values = []
    for p in sample:
        try:
            arr = np.array(Image.open(p).convert("RGB").resize((img_size[1], img_size[0])), dtype=np.float32) / 255.0
            pixel_values.append(arr)
        except Exception:
            pass
    if pixel_values:
        stacked = np.stack(pixel_values)
        logger.info(
            "Dataset stats (sample=%d) — total=%d | mean=%.4f | std=%.4f",
            sample_size,
            len(paths),
            stacked.mean(),
            stacked.std(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo data generator — runs with ZERO external dependencies
# ──────────────────────────────────────────────────────────────────────────────

def generate_demo_data(config_path: str = "config.yaml") -> None:
    """Generate synthetic normal and defective images for a fully offline demo.

    Normal images are smooth Gaussian-blurred colour blobs simulating a uniform
    manufactured surface. Defective images are identical but with a rectangular
    high-contrast patch inserted to simulate a scratch, stain, or crack.

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg = load_config(config_path)
    demo_cfg = cfg["data"]["demo"]
    paths_cfg = cfg["paths"]

    img_h, img_w = demo_cfg["img_size"]
    num_normal = demo_cfg["num_normal"]
    num_defect = demo_cfg["num_defect"]
    noise_std = demo_cfg["noise_std"]
    ph, pw = demo_cfg["defect_patch_size"]

    normal_dir = Path(paths_cfg["normal_data"])
    defect_dir = Path(paths_cfg["defect_data"])
    normal_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    logger.info("Generating %d normal demo images …", num_normal)
    for i in range(num_normal):
        img = _make_normal_image(img_h, img_w, rng, noise_std)
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil.save(normal_dir / f"normal_{i:04d}.png")

    logger.info("Generating %d defective demo images …", num_defect)
    for i in range(num_defect):
        img = _make_normal_image(img_h, img_w, rng, noise_std)
        img = _insert_defect(img, rng, ph, pw)
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil.save(defect_dir / f"defect_{i:04d}.png")

    logger.info(
        "Demo data ready — normal: %s | defect: %s",
        normal_dir,
        defect_dir,
    )


def _make_normal_image(h: int, w: int, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    """Create a synthetic normal surface image.

    Generates a smooth gradient field per channel with mild Gaussian noise,
    simulating a uniform textured surface.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        rng: NumPy random generator instance.
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        Float32 ndarray of shape (h, w, 3) in [0, 1].
    """
    base_color = rng.uniform(0.3, 0.75, size=(3,))
    img = np.ones((h, w, 3), dtype=np.float32) * base_color

    # Add smooth gradient variation
    gy = np.linspace(-0.1, 0.1, h)[:, None, None]
    gx = np.linspace(-0.1, 0.1, w)[None, :, None]
    img = img + gy + gx

    # Add slight per-channel texture
    for c in range(3):
        freq = rng.integers(3, 8)
        phase = rng.uniform(0, np.pi)
        texture_h = 0.03 * np.sin(np.linspace(0, freq * np.pi + phase, h))[:, None]
        texture_w = 0.03 * np.sin(np.linspace(0, freq * np.pi + phase, w))[None, :]
        img[:, :, c] += texture_h + texture_w

    # Add Gaussian noise
    img += rng.normal(0, noise_std, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _insert_defect(
    img: np.ndarray,
    rng: np.random.Generator,
    patch_h: int,
    patch_w: int,
) -> np.ndarray:
    """Insert a rectangular defect patch onto an image.

    The patch simulates anomalies such as scratches or stains by placing a
    high-contrast rectangular region at a random location.

    Args:
        img: Base float32 image of shape (H, W, 3).
        rng: NumPy random generator instance.
        patch_h: Height of the defect patch.
        patch_w: Width of the defect patch.

    Returns:
        Modified float32 image with defect inserted.
    """
    h, w, _ = img.shape
    img = img.copy()

    y0 = rng.integers(0, max(1, h - patch_h))
    x0 = rng.integers(0, max(1, w - patch_w))
    y1 = min(y0 + patch_h, h)
    x1 = min(x0 + patch_w, w)

    defect_type = rng.integers(0, 3)

    if defect_type == 0:
        # Bright scratch-like patch
        img[y0:y1, x0:x1, :] = rng.uniform(0.85, 1.0, size=(y1 - y0, x1 - x0, 3)).astype(np.float32)
    elif defect_type == 1:
        # Dark stain
        img[y0:y1, x0:x1, :] = rng.uniform(0.0, 0.15, size=(y1 - y0, x1 - x0, 3)).astype(np.float32)
    else:
        # Colour-shifted blob
        shift = rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
        img[y0:y1, x0:x1, :] = np.clip(img[y0:y1, x0:x1, :] + shift, 0.0, 1.0)

    return img
