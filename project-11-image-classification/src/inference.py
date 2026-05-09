"""
inference.py
------------
Production inference pipeline for loading a trained model and
predicting on single images or batches.

Can be run from the command line:
    python -m src.inference --image path/to/image.jpg --model transfer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from src.config import (
    BASELINE_MODEL_PATH,
    CIFAR10_CLASSES,
    IMAGE_SIZE,
    IMAGE_SIZE_TL,
    CIFAR10_MEAN,
    CIFAR10_STD,
    TRANSFER_MODEL_PATH,
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_type: str = "transfer") -> keras.Model:
    path = BASELINE_MODEL_PATH if model_type == "baseline" else TRANSFER_MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run train.py first to generate trained models."
        )
    print(f"[INFO] Loading model from {path}")
    
    # Strategy 1: Try standard load with empty custom_objects
    try:
        model = keras.models.load_model(path, compile=False, custom_objects={})
        print(f"[INFO] Successfully loaded model using standard method")
        return model
    except (ValueError, TypeError, AttributeError) as e:
        print(f"[WARN] Standard load failed: {e}")
    
    # Strategy 2: Try loading via h5py with config reconstruction
    try:
        import h5py
        from tensorflow.keras.models import model_from_json
        
        print("[INFO] Attempting to load model via h5py...")
        with h5py.File(path, "r") as f:
            # Get model config
            model_config = f.attrs.get("model_config")
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")
            
            # Try to fix common serialization issues
            model_config = model_config.replace(
                '"batch_shape"', '"batch_input_shape"'
            )
            
            # Create model from config
            model = model_from_json(model_config, custom_objects={})
            model.load_weights(path)
            print(f"[INFO] Successfully loaded model using h5py method")
            return model
    except Exception as e:
        print(f"[WARN] h5py method failed: {e}")
    
    # Strategy 3: Try with safe_mode=False (Keras 3.x)
    try:
        print("[INFO] Attempting to load with safe_mode=False...")
        model = keras.models.load_model(
            path, 
            compile=False, 
            custom_objects={},
            safe_mode=False
        )
        print(f"[INFO] Successfully loaded model with safe_mode=False")
        return model
    except TypeError:
        # safe_mode parameter not supported in this version
        pass
    except Exception as e:
        print(f"[WARN] safe_mode method failed: {e}")
    
    # If all strategies fail, raise informative error
    raise RuntimeError(
        f"Failed to load model from {path}. This can happen due to:\n"
        "1. TensorFlow/Keras version mismatch (model was saved with a different version)\n"
        "2. Corrupted model file\n"
        "3. Missing custom layers or objects\n\n"
        "Solution: Retrain the model by running: python train.py"
    )
    return model

# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    model_type: str = "transfer",
) -> np.ndarray:
    """
    Load and preprocess a single image for inference.

    Accepts: file path, numpy array (H×W×3 uint8), or PIL Image.
    Returns: float32 array of shape (1, H, W, 3), ready for model.predict().
    """
    target_size = IMAGE_SIZE_TL if model_type == "transfer" else IMAGE_SIZE

    # Load from path or convert from PIL
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    pil_img  = pil_img.resize(target_size, Image.BILINEAR)
    img_arr  = np.array(pil_img, dtype=np.float32) / 255.0

    # Channel-wise standardisation
    mean = np.array(CIFAR10_MEAN, dtype=np.float32)
    std  = np.array(CIFAR10_STD,  dtype=np.float32)
    img_arr = (img_arr - mean) / std

    return np.expand_dims(img_arr, 0)          # (1, H, W, 3)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(
    model:      keras.Model,
    image:      Union[str, Path, np.ndarray, Image.Image],
    model_type: str = "transfer",
    top_k:      int = 3,
) -> list[dict]:
    """
    Run inference on a single image.

    Args:
        model:      Loaded Keras model.
        image:      Input image (see preprocess_image for accepted types).
        model_type: "baseline" or "transfer" (determines resize target).
        top_k:      Number of top predictions to return.

    Returns:
        List of dicts sorted by confidence (descending):
        [{"class": "cat", "label_idx": 3, "confidence": 0.87}, ...]
    """
    x      = preprocess_image(image, model_type=model_type)
    proba  = model.predict(x, verbose=0)[0]

    top_indices = np.argsort(proba)[::-1][:top_k]
    results = [
        {
            "class":      CIFAR10_CLASSES[i],
            "label_idx":  int(i),
            "confidence": float(proba[i]),
        }
        for i in top_indices
    ]
    return results


def predict_batch(
    model:      keras.Model,
    images:     list,
    model_type: str = "transfer",
) -> list[dict]:
    """
    Run inference on a list of images in a single forward pass.

    Returns:
        List of top-1 prediction dicts (one per image).
    """
    xs     = np.concatenate(
        [preprocess_image(img, model_type=model_type) for img in images], axis=0
    )
    proba  = model.predict(xs, verbose=0)
    top1   = np.argmax(proba, axis=1)

    return [
        {
            "class":      CIFAR10_CLASSES[i],
            "label_idx":  int(i),
            "confidence": float(proba[idx, i]),
        }
        for idx, i in enumerate(top1)
    ]


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 inference script")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--model",  default="transfer", choices=["baseline", "transfer"],
                        help="Which model to use (default: transfer)")
    parser.add_argument("--top_k",  type=int, default=3,
                        help="Number of top predictions to display")
    args = parser.parse_args()

    model   = load_model(args.model)
    results = predict(model, args.image, model_type=args.model, top_k=args.top_k)

    print(f"\nPredictions for: {args.image}")
    print("─" * 40)
    for rank, r in enumerate(results, 1):
        bar = "█" * int(r["confidence"] * 30)
        print(f"  {rank}. {r['class']:<12} {r['confidence']*100:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    _cli()
