"""
models.py
---------
Model factory functions for:
    A) Baseline CNN  — built from scratch on 32×32 CIFAR-10 images.
    B) Transfer Learning model — MobileNetV2 pretrained on ImageNet,
       fine-tuned on CIFAR-10.

Why Transfer Learning outperforms the baseline:
    MobileNetV2 has learned rich, hierarchical feature detectors (edges,
    textures, object parts) from 1.2 M ImageNet images.  Even though CIFAR-10
    is low-resolution, these generalised features transfer well and provide a
    far better initialisation than random weights.  This lets us reach higher
    accuracy with fewer epochs and less risk of overfitting.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    IMAGE_SIZE,
    IMAGE_SIZE_TL,
    LEARNING_RATE,
    NUM_CHANNELS,
    NUM_CLASSES,
    TL_FINE_TUNE_LR,
)


# ── A) Baseline CNN ──────────────────────────────────────────────────────────

def build_baseline_cnn(
    input_shape: tuple = (*IMAGE_SIZE, NUM_CHANNELS),
    num_classes: int   = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """
    A simple but well-regularised convolutional network.

    Architecture rationale:
        - Two conv blocks (Conv → BN → ReLU → Pool) extract local features
          at two levels of abstraction.
        - Batch Normalisation after each conv stabilises activations and
          acts as a mild regulariser.
        - Dropout(0.5) before the dense head prevents co-adaptation of neurons.
        - Softmax output maps logits to a probability distribution.

    This model intentionally avoids residual connections and attention
    mechanisms so we have a clear apples-to-apples baseline.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # ── Block 1 ────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Block 2 ────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ── Block 3 ────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # ── Classifier head ────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.5, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="BaselineCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── B) Transfer Learning Model (MobileNetV2) ─────────────────────────────────

def build_transfer_model(
    input_shape: tuple   = (*IMAGE_SIZE_TL, NUM_CHANNELS),
    num_classes: int     = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    trainable_layers: int = 30,        # Number of top MobileNetV2 layers to unfreeze
) -> keras.Model:
    """
    MobileNetV2 feature extractor + custom classification head, with optional
    fine-tuning of the top layers of the backbone.

    Why MobileNetV2?
        - Lightweight (3.4 M params) → fast to fine-tune on commodity hardware.
        - Inverted residuals + depthwise separable convolutions → efficient.
        - Strong ImageNet accuracy despite small footprint.
        - Ships with Keras → no extra dependencies.

    Two-phase training strategy:
        Phase 1 — Feature extraction: freeze the entire backbone, train only
                  the new classification head.  This prevents large gradients
                  from the randomly-initialised head corrupting pretrained weights.
        Phase 2 — Fine-tuning: unfreeze the top `trainable_layers` and resume
                  training with a much lower learning rate.

    Call `freeze_backbone()` / `unfreeze_top(n)` helpers below to switch phases.
    """
    # Pretrained backbone — weights from ImageNet, top FC layer excluded
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False    # Freeze all backbone layers initially

    inputs = keras.Input(shape=input_shape, name="image_input")

    # MobileNetV2 expects inputs pre-processed to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)       # training=False keeps BN in inference mode

    # ── Custom classification head ──────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.4, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="TransferMobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Attach the base_model reference so callers can unfreeze later
    model._base_model = base_model
    return model


def unfreeze_top_layers(model: keras.Model, n_layers: int, fine_tune_lr: float = TL_FINE_TUNE_LR) -> None:
    """
    Unfreeze the top `n_layers` of the MobileNetV2 backbone for fine-tuning.
    Re-compiles with a lower learning rate to avoid destroying pretrained representations.

    Args:
        model:        The transfer learning model returned by build_transfer_model().
        n_layers:     How many layers from the top of the backbone to unfreeze.
        fine_tune_lr: Learning rate for fine-tuning phase.
    """
    base = model._base_model
    base.trainable = True

    # Keep all layers except the top n frozen
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[INFO] Unfroze top {n_layers} backbone layers. Fine-tune LR = {fine_tune_lr}")
