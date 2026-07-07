"""
Basic sanity tests for the model architecture and fine-tuning logic.
Uses weights=None so tests run fully offline (no ImageNet weight download).

Run with: pytest tests/
"""

import numpy as np

from src.config import IMG_SHAPE
from src.model import build_model, enable_fine_tuning


def test_build_model_output_shape():
    model, base_model = build_model(weights=None)
    dummy_input = np.zeros((1,) + IMG_SHAPE, dtype="float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (1, 1)


def test_build_model_backbone_frozen_initially():
    model, base_model = build_model(weights=None)
    assert base_model.trainable is False


def test_output_is_valid_probability():
    model, base_model = build_model(weights=None)
    dummy_input = np.random.rand(2, *IMG_SHAPE).astype("float32") * 255
    output = model.predict(dummy_input, verbose=0)
    assert (output >= 0).all() and (output <= 1).all()


def test_enable_fine_tuning_unfreezes_top_layers():
    model, base_model = build_model(weights=None)
    enable_fine_tuning(model, base_model)
    assert base_model.trainable is True
    # Early layers should remain frozen, later layers should be trainable
    assert base_model.layers[0].trainable is False
    assert base_model.layers[-1].trainable is True
