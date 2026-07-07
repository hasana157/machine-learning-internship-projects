"""
Dataset download and loading utilities for the Cats vs Dogs dataset.

Fixes a subtle bug from the original single-file script: the raw download
already ships separate `train/` and `validation/` folders, so we point
`image_dataset_from_directory` at each directly instead of re-splitting the
`train/` folder with `validation_split` (which would train and validate on
folders that were never designed for that split, and would silently break
on a second run since the rename-if-not-exists logic is not idempotent).
"""

from pathlib import Path

import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    DATASET_ARCHIVE_NAME,
    DATASET_URL,
    EXTRACTED_FOLDER_NAME,
    IMG_SIZE,
    RAW_DATA_DIR,
)


def download_dataset() -> Path:
    """
    Download and extract the Cats vs Dogs dataset (idempotent — skips
    re-downloading if it's already present). Returns the path to the
    extracted folder containing `train/` and `validation/` subfolders.
    """
    archive_path = tf.keras.utils.get_file(
        DATASET_ARCHIVE_NAME,
        origin=DATASET_URL,
        extract=True,
        cache_dir=str(RAW_DATA_DIR.parent),  # keep everything under data/
        cache_subdir="raw",
    )
    extracted_dir = RAW_DATA_DIR / EXTRACTED_FOLDER_NAME
    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted dataset at {extracted_dir}, but it wasn't found. "
            f"Archive was placed at {archive_path}."
        )
    return extracted_dir


def load_datasets():
    """
    Download (if needed) the dataset and return (train_ds, val_ds) as
    tf.data.Dataset objects, ready for model.fit(). Applies caching and
    prefetching for faster training.
    """
    extracted_dir = download_dataset()
    train_dir = extracted_dir / "train"
    val_dir = extracted_dir / "validation"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names  # e.g. ["cats", "dogs"]

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names
