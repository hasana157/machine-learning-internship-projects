"""
Run inference on custom cat/dog images using the trained model.

Usage:
    python -m src.predict

Custom images should be placed in data/custom_images/ as .png/.jpg/.jpeg files.
Note: preprocessing (MobileNetV2's preprocess_input) is baked into the model
itself (see src/model.py), so this script only needs to resize + batch the
image — no manual normalization needed, and no risk of mismatched
preprocessing between training and inference.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASS_NAMES, CUSTOM_IMAGES_DIR, IMG_SIZE, MODEL_PATH


def load_trained_model():
    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}.")
        print("Train one first with:  python -m src.train")
        sys.exit(1)
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image_path(img_path) -> np.ndarray:
    """Load an image, resize to IMG_SIZE, and batch it as (1, H, W, 3)."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32")
    return np.expand_dims(img_array, axis=0)


def predict_image(model, img_path):
    """Return (label, confidence, raw_display_array) for one image."""
    img_array = preprocess_image_path(img_path)
    prob = float(model.predict(img_array, verbose=0)[0][0])  # P(class == "dog")
    label = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence, img_array[0].astype("uint8")


def main():
    model = load_trained_model()

    image_paths = sorted(
        p
        for ext in ("*.png", "*.jpg", "*.jpeg")
        for p in CUSTOM_IMAGES_DIR.glob(ext)
    )

    if not image_paths:
        print(f"No images found in {CUSTOM_IMAGES_DIR}")
        print("Add some .png/.jpg cat or dog photos there and re-run.")
        return

    n = len(image_paths)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, img_path in zip(axes, image_paths):
        label, confidence, display_img = predict_image(model, img_path)
        print(f"{img_path.name}: predicted={label}  confidence={confidence:.2%}")

        ax.imshow(display_img)
        ax.set_title(f"{img_path.name}\nPred: {label} ({confidence:.0%})")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
