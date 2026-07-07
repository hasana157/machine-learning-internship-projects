"""
Model architecture definition, kept separate from training so both
train.py and any future experiments (or the notebook) can import the
exact same architecture without copy-pasting code.
"""

import tensorflow as tf


def build_model() -> tf.keras.Model:
    """
    Build and compile a simple fully-connected neural network for
    MNIST digit classification (28x28 grayscale -> 10 classes).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
