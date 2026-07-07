"""
Run inference on custom handwritten digit images using the trained model.

Usage:
    python -m src.predict

Custom images should be placed in data/custom_digits/ as .png/.jpg files.
They can be any size/color — preprocessing handles grayscale conversion,
resizing to 28x28, and inversion automatically (see src/preprocessing.py).
"""

import sys

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import CUSTOM_DIGITS_DIR, MODEL_PATH
from src.preprocessing import preprocess_image_path


def load_trained_model():
    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}.")
        print("Train one first with:  python -m src.train")
        sys.exit(1)
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(model, img_path):
    """Return (predicted_digit, confidence, preprocessed_array) for one image."""
    img_array = preprocess_image_path(img_path)
    pred = model.predict(img_array, verbose=0)
    digit = int(pred.argmax())
    confidence = float(pred.max())
    return digit, confidence, img_array


def main():
    model = load_trained_model()

    image_paths = sorted(
        list(CUSTOM_DIGITS_DIR.glob("*.png")) + list(CUSTOM_DIGITS_DIR.glob("*.jpg"))
    )

    if not image_paths:
        print(f"No images found in {CUSTOM_DIGITS_DIR}")
        print("Add some .png/.jpg handwritten digit images there and re-run.")
        return

    n = len(image_paths)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, img_path in zip(axes, image_paths):
        digit, confidence, img_array = predict_image(model, img_path)
        print(f"{img_path.name}: predicted={digit}  confidence={confidence:.2%}")

        ax.imshow(img_array.reshape(28, 28), cmap="gray")
        ax.set_title(f"{img_path.name}\nPred: {digit} ({confidence:.0%})")
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
