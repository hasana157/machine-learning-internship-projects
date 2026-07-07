"""
Shared configuration and paths for the Cat vs Dog transfer-learning project.
Centralizing these here means data.py, model.py, train.py, predict.py, the
notebook, and the Streamlit app all stay in sync.
"""

from pathlib import Path

# Project root = parent of the "src" folder this file lives in
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # where the extracted dataset lives
CUSTOM_IMAGES_DIR = DATA_DIR / "custom_images"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "cat_dog_transfer.h5"
TRAINING_CURVE_PATH = FIGURES_DIR / "training_curve.png"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Dataset
DATASET_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
DATASET_ARCHIVE_NAME = "cats_and_dogs_filtered.zip"
# After extraction, tf.keras caches it under ~/.keras/datasets/ by default;
# we then point our own RAW_DATA_DIR-relative TRAIN/VAL dirs at it.
EXTRACTED_FOLDER_NAME = "cats_and_dogs_filtered"

# Model / training hyperparameters
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32
HEAD_EPOCHS = 5          # phase 1: train classification head, backbone frozen
FINE_TUNE_EPOCHS = 5     # phase 2: fine-tune top backbone layers
FINE_TUNE_AT_LAYER = -50  # unfreeze only the last N layers of the backbone
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

CLASS_NAMES = ["cat", "dog"]  # alphabetical order, matches image_dataset_from_directory

# Make sure key directories exist whenever this config is imported
for _dir in (DATA_DIR, RAW_DATA_DIR, CUSTOM_IMAGES_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
