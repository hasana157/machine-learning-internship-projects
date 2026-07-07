"""
Model architecture for transfer learning on Cats vs Dogs, kept separate
from training so train.py, predict.py, the notebook, and the Streamlit
app all build/compile the model identically.
"""

import tensorflow as tf

from src.config import (
    FINE_TUNE_AT_LAYER,
    FINE_TUNE_LEARNING_RATE,
    HEAD_LEARNING_RATE,
    IMG_SHAPE,
)


def build_model(weights: str = "imagenet"):
    """
    Build a MobileNetV2-backed binary classifier.

    Args:
        weights: "imagenet" for pretrained weights (needs internet on first
            run), or None to build the architecture without downloading
            weights (useful for offline architecture/shape tests).

    Returns:
        (model, base_model) — base_model is returned separately so the
        caller can freeze/unfreeze it for fine-tuning.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False  # freeze backbone for phase 1

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    compile_head(model)
    return model, base_model


def compile_head(model):
    """Compile for phase 1: train the classification head only."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(HEAD_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def enable_fine_tuning(model, base_model):
    """
    Unfreeze the top layers of the backbone for phase 2 (fine-tuning) and
    recompile with a much lower learning rate so we don't destroy the
    pretrained weights.
    """
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
