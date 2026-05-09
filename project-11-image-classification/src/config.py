"""
config.py
---------
Central configuration file for all project constants, paths, and hyperparameters.
Keeping these in one place makes the project easy to tune without hunting through files.
"""

import os
from pathlib import Path

# ── Project root (two levels up from this file) ────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Directory paths ─────────────────────────────────────────────────────────
MODELS_DIR      = ROOT_DIR / "models"
FIGURES_DIR     = ROOT_DIR / "reports" / "figures"
NOTEBOOKS_DIR   = ROOT_DIR / "notebooks"

# Auto-create directories if they don't exist
for _dir in [MODELS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Dataset ─────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
NUM_CLASSES     = len(CIFAR10_CLASSES)
IMAGE_SIZE      = (32, 32)          # Native CIFAR-10 resolution
IMAGE_SIZE_TL   = (96, 96)         # Upsampled for transfer learning (MobileNetV2 min: 96)
NUM_CHANNELS    = 3

# ── Training hyperparameters ────────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS_BASELINE = 40
EPOCHS_TL       = 30               # Transfer learning (faster convergence)
LEARNING_RATE   = 1e-3
TL_FINE_TUNE_LR = 1e-5             # Lower LR for fine-tuning pretrained layers
VALIDATION_SPLIT = 0.15

# ── Normalisation stats (CIFAR-10 channel-wise mean & std) ──────────────────
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

# ── Model save paths ────────────────────────────────────────────────────────
BASELINE_MODEL_PATH  = str(MODELS_DIR / "baseline_model.h5")
TRANSFER_MODEL_PATH  = str(MODELS_DIR / "transfer_model.h5")

# ── Early stopping ──────────────────────────────────────────────────────────
PATIENCE = 8                        # Epochs with no improvement before stopping

# ── Random seed ─────────────────────────────────────────────────────────────
SEED = 42
