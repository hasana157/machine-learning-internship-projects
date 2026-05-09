"""
data_utils.py — Dataset loading, augmentation, and preprocessing utilities.

Designed for the Kaggle Masked Face Recognition dataset:
https://www.kaggle.com/datasets/muhammeddalkran/masked-facerecognition

Dataset limitation: Classification-only (mask/ and no_mask/ folders).
No bounding box annotations — overcome via YOLO-based face detection in pipeline.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE     = 128
BATCH_SIZE   = 32
RANDOM_SEED  = 42
CLASS_NAMES  = ["mask", "no_mask"]


# ─── Dataset Loader ───────────────────────────────────────────────────────────
def get_data_generators(
    data_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.2,
    augment: bool = True,
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, ...]:
    """
    Load train/validation datasets with optional augmentation.

    Args:
        data_dir: Root folder with mask/ and no_mask/ subdirectories.
        img_size:  Resize target for each image.
        batch_size: Training batch size.
        val_split:  Fraction reserved for validation.
        augment:    Apply data augmentation to training set.

    Returns:
        (train_gen, val_gen, class_indices)
    """
    target_size = (img_size, img_size)

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            shear_range=10,
            fill_mode="nearest",
            validation_split=val_split,
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=val_split,
        )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        seed=RANDOM_SEED,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=False,
    )

    return train_gen, val_gen, train_gen.class_indices


def get_test_generator(
    data_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
) -> tf.keras.preprocessing.image.DirectoryIterator:
    """Load test set (no augmentation, no shuffle)."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )


# ─── Preprocessing for Inference ──────────────────────────────────────────────
def preprocess_face(face_bgr: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Preprocess a raw BGR face crop for CNN inference.
    Returns float32 array of shape (1, img_size, img_size, 3).
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (img_size, img_size))
    face_norm = face_resized.astype(np.float32) / 255.0
    return np.expand_dims(face_norm, axis=0)


# ─── Class Distribution ───────────────────────────────────────────────────────
def get_class_distribution(data_dir: str) -> Dict[str, int]:
    """Count images per class in data_dir."""
    dist = {}
    data_path = Path(data_dir)
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")) +
                        list(class_dir.glob("*.jpeg")) +
                        list(class_dir.glob("*.png")))
            dist[class_dir.name] = count
    return dist


def load_sample_images(
    data_dir: str,
    n_per_class: int = 5,
    img_size: int = IMG_SIZE,
) -> Dict[str, List[np.ndarray]]:
    """Load a few sample images per class for EDA."""
    samples = {}
    data_path = Path(data_dir)
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        imgs = []
        for p in list(class_dir.glob("*.jpg"))[:n_per_class]:
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            imgs.append(img)
        samples[class_dir.name] = imgs
    return samples
