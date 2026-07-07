"""
Shared configuration and paths for the MNIST digit classifier project.
Centralizing paths here means train.py, predict.py, the notebook, and the
Streamlit app all stay in sync if the project is moved or renamed.
"""

from pathlib import Path

# Project root = parent of the "src" folder this file lives in
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
CUSTOM_DIGITS_DIR = DATA_DIR / "custom_digits"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "mnist_digit_model.h5"

IMG_SIZE = (28, 28)
EPOCHS = 5
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# Make sure key directories exist whenever this config is imported
for _dir in (DATA_DIR, CUSTOM_DIGITS_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
