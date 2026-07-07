"""
model.py
========
Convolutional Autoencoder architecture for VisualSentry.

The model learns a compressed latent representation of normal surface images.
At inference time, defective images produce high reconstruction error because
the decoder has only learned to reconstruct normal patterns.
"""

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ConvAutoencoder(tf.keras.Model):
    """Convolutional Autoencoder for unsupervised anomaly detection.

    The encoder progressively down-samples the input through strided convolutions
    and max-pooling, projecting to a compact latent vector. The decoder mirrors
    this structure with transposed convolutions to reconstruct the original image.
    High reconstruction error at inference time signals an anomaly.

    Attributes:
        img_size: Spatial dimensions (height, width) of input images.
        channels: Number of input colour channels.
        latent_dim: Dimensionality of the bottleneck latent space.
        encoder_filters: List of filter counts for successive encoder Conv2D layers.
        decoder_filters: List of filter counts for successive decoder Conv2DTranspose layers.
        _encoder_model: Keras functional sub-model exposing the latent space.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        channels: int = 3,
        latent_dim: int = 64,
        encoder_filters: List[int] = (32, 64, 128),
        decoder_filters: List[int] = (128, 64, 32),
        activation: str = "relu",
        output_activation: str = "sigmoid",
        **kwargs,
    ) -> None:
        """Initialise the ConvAutoencoder.

        Args:
            img_size: (height, width) of input images.
            channels: Number of channels in input images (3 for RGB).
            latent_dim: Size of the bottleneck dense layer.
            encoder_filters: Convolutional filter sizes for the encoder path.
            decoder_filters: Convolutional filter sizes for the decoder path.
            activation: Activation function for intermediate layers.
            output_activation: Activation function for the final reconstruction layer.
        """
        super().__init__(**kwargs)

        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.encoder_filters = list(encoder_filters)
        self.decoder_filters = list(decoder_filters)
        self.activation = activation
        self.output_activation = output_activation

        # Compute the spatial size after max-pooling through all encoder stages
        self._pool_factor = 2 ** len(self.encoder_filters)
        self._bottleneck_h = img_size[0] // self._pool_factor
        self._bottleneck_w = img_size[1] // self._pool_factor
        self._bottleneck_channels = self.encoder_filters[-1]
        self._bottleneck_units = (
            self._bottleneck_h * self._bottleneck_w * self._bottleneck_channels
        )

        # ── Encoder layers ────────────────────────────────────────────────────
        self.enc_convs = []
        self.enc_pools = []
        for f in self.encoder_filters:
            self.enc_convs.append(
                layers.Conv2D(f, 3, activation=activation, padding="same")
            )
            self.enc_pools.append(layers.MaxPooling2D(2, padding="same"))

        self.flatten = layers.Flatten()
        self.dense_latent = layers.Dense(latent_dim, name="latent_space")

        # ── Decoder layers ────────────────────────────────────────────────────
        self.dense_decode = layers.Dense(self._bottleneck_units, activation=activation)
        self.reshape = layers.Reshape(
            (self._bottleneck_h, self._bottleneck_w, self._bottleneck_channels)
        )

        self.dec_convT = []
        for f in self.decoder_filters:
            self.dec_convT.append(
                layers.Conv2DTranspose(f, 3, strides=2, activation=activation, padding="same")
            )

        self.output_conv = layers.Conv2D(
            channels, 3, activation=output_activation, padding="same", name="reconstruction"
        )

        # Build the functional encoder sub-model for latent extraction
        self._encoder_model: tf.keras.Model = self._build_encoder_model()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Full forward pass: encode → decode → reconstruct.

        Args:
            inputs: Batch of images, shape (B, H, W, C), float32 in [0, 1].
            training: Whether the model is in training mode (affects BatchNorm/Dropout).

        Returns:
            Reconstructed images of the same shape as inputs.
        """
        z = self._encode(inputs)
        return self._decode(z)

    def _encode(self, x: tf.Tensor) -> tf.Tensor:
        """Run the encoder path to produce a latent vector.

        Args:
            x: Input image batch.

        Returns:
            Latent representation tensor of shape (B, latent_dim).
        """
        for conv, pool in zip(self.enc_convs, self.enc_pools):
            x = conv(x)
            x = pool(x)
        x = self.flatten(x)
        return self.dense_latent(x)

    def _decode(self, z: tf.Tensor) -> tf.Tensor:
        """Run the decoder path to reconstruct from a latent vector.

        Args:
            z: Latent tensor of shape (B, latent_dim).

        Returns:
            Reconstructed image batch.
        """
        x = self.dense_decode(z)
        x = self.reshape(x)
        for convT in self.dec_convT:
            x = convT(x)
        return self.output_conv(x)

    # ── Sub-model access ──────────────────────────────────────────────────────

    def get_encoder(self) -> tf.keras.Model:
        """Return the encoder as a standalone Keras model.

        This can be used to extract latent representations for downstream tasks
        such as clustering, visualisation with t-SNE/UMAP, or nearest-neighbour
        anomaly detection.

        Returns:
            Keras Model mapping (H, W, C) images to latent vectors.
        """
        return self._encoder_model

    def _build_encoder_model(self) -> tf.keras.Model:
        """Construct a functional Keras Model for the encoder sub-graph.

        Returns:
            Functional model with the same encoder weights as this autoencoder.
        """
        inp = tf.keras.Input(shape=(self.img_size[0], self.img_size[1], self.channels))
        x = inp
        for conv, pool in zip(self.enc_convs, self.enc_pools):
            x = conv(x)
            x = pool(x)
        x = self.flatten(x)
        z = self.dense_latent(x)
        return tf.keras.Model(inputs=inp, outputs=z, name="encoder")

    # ── Utility ───────────────────────────────────────────────────────────────

    def build_graph(self) -> None:
        """Force-build the model graph so that model.summary() is available.

        Call this once before training or before calling summary().
        """
        dummy = tf.keras.Input(
            shape=(self.img_size[0], self.img_size[1], self.channels)
        )
        self(dummy)

    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml") -> "ConvAutoencoder":
        """Instantiate a ConvAutoencoder directly from a YAML config file.

        Args:
            config_path: Path to the YAML configuration.

        Returns:
            Configured ConvAutoencoder instance.
        """
        cfg = load_config(config_path)["model"]
        return cls(
            img_size=tuple(cfg["img_size"]),
            channels=cfg.get("channels", 3),
            latent_dim=cfg["latent_dim"],
            encoder_filters=cfg["encoder_filters"],
            decoder_filters=cfg.get("decoder_filters", list(reversed(cfg["encoder_filters"]))),
            activation=cfg.get("activation", "relu"),
            output_activation=cfg.get("output_activation", "sigmoid"),
        )
