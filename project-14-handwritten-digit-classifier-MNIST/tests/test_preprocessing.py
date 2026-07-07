"""
Basic sanity tests for the preprocessing pipeline.
Run with: pytest tests/
"""

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_pil_image


def test_output_shape():
    img = Image.new("L", (100, 100), color=0)
    result = preprocess_pil_image(img)
    assert result.shape == (1, 28, 28)


def test_output_range():
    img = Image.new("L", (50, 50), color=200)
    result = preprocess_pil_image(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_auto_invert_flips_bright_background():
    # Bright background, dark "digit" pixel in the corner
    arr = np.full((28, 28), 255, dtype=np.uint8)
    arr[0, 0] = 0
    img = Image.fromarray(arr, mode="L")

    result = preprocess_pil_image(img, auto_invert=True)
    # After inversion, background pixels should be near 0 (dark)
    assert result[0, -1, -1] < 0.5


def test_dtype_is_float():
    img = Image.new("L", (28, 28), color=128)
    result = preprocess_pil_image(img)
    assert result.dtype == np.float32
