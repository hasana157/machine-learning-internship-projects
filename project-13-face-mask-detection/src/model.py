"""
model.py — CNN architecture for mask/no-mask binary classification.

Design philosophy:
- MobileNetV2 transfer-learned backbone (fast + accurate for edge deployment)
- Custom classification head with dropout for regularization
- Input: 128×128×3 normalized float images
- Output: sigmoid probability (1 = mask, 0 = no_mask)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


# ─── Transfer-Learning Model (Primary) ────────────────────────────────────────
def build_classifier(
    img_size: int = 128,
    dropout_rate: float = 0.4,
    freeze_base: bool = True,
) -> Model:
    """
    MobileNetV2 backbone + custom head for mask classification.

    Args:
        img_size:      Input image size (square).
        dropout_rate:  Dropout rate for regularization.
        freeze_base:   Whether to freeze base weights initially.

    Returns:
        Compiled Keras model.
    """
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="MaskClassifier_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def build_custom_cnn(img_size: int = 128, dropout_rate: float = 0.4) -> Model:
    """
    Lightweight custom CNN (no transfer learning) — used for augmentation comparison.

    Architecture: 3× [Conv→BN→ReLU→MaxPool] → GlobalAvgPool → Dense → Sigmoid
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="MaskClassifier_CustomCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def unfreeze_top_layers(model: Model, n_layers: int = 30) -> Model:
    """
    Fine-tune: Unfreeze the top N layers of the base model for stage-2 training.
    """
    base = model.layers[1]          # MobileNetV2 is layer index 1
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_callbacks(model_path: str = "models/mask_classifier.keras"):
    """Standard callbacks: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau."""
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="reports/logs",
            histogram_freq=1,
        ),
    ]
