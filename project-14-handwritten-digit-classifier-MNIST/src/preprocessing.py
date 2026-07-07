"""
Shared preprocessing utilities.

Used by src/predict.py, the notebook, and the Streamlit app so all three
apply IDENTICAL preprocessing to custom images before they hit the model.
Mismatched preprocessing between training and inference is the #1 cause
of "my custom digit predictions are wrong" bugs.
"""

import numpy as np
from PIL import Image, ImageOps

from src.config import IMG_SIZE


def preprocess_pil_image(img: Image.Image, auto_invert: bool = True) -> np.ndarray:
    """
    Convert a PIL image into a (1, 28, 28) float32 array ready for the model.

    Steps:
      1. Convert to grayscale ("L" mode)
      2. Resize to 28x28 (MNIST's native resolution)
      3. Optionally auto-invert so the digit is white-on-black, matching
         MNIST's convention (most phone photos/scans are black-on-white)
      4. Normalize pixel values to [0, 1]
      5. Reshape to (1, 28, 28) for model.predict()

    Args:
        img: A PIL Image, any mode/size.
        auto_invert: If True, automatically flips colors when the image
            appears to be dark-digit-on-light-background.

    Returns:
        np.ndarray of shape (1, 28, 28), dtype float32.
    """
    img = img.convert("L")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img).astype("float32")

    if auto_invert:
        # If the average pixel is bright (light background), MNIST expects
        # the opposite (dark background, bright digit) — so invert.
        if img_array.mean() > 127:
            img_array = 255.0 - img_array

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1])
    return img_array


def preprocess_image_path(img_path) -> np.ndarray:
    """Load an image from disk and preprocess it. See preprocess_pil_image."""
    img = Image.open(img_path)
    return preprocess_pil_image(img)
