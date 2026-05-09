"""
test_models.py
--------------
Unit tests for model construction, data pipeline, and inference.
Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGE_SIZE, IMAGE_SIZE_TL, NUM_CLASSES, BATCH_SIZE
from src.data_loader import normalise, split_train_val
from src.models import build_baseline_cnn, build_transfer_model


# ── Normalisation ─────────────────────────────────────────────────────────────

class TestNormalisation:
    def test_output_range(self):
        x = np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
        x_norm = normalise(x)
        # After channel-wise standardisation, values will be roughly in [-3, 3]
        assert x_norm.min() > -10
        assert x_norm.max() <  10

    def test_dtype(self):
        x = np.ones((10, 32, 32, 3), dtype=np.uint8) * 128
        x_norm = normalise(x)
        assert x_norm.dtype == np.float32

    def test_shape_preserved(self):
        x = np.zeros((5, 32, 32, 3), dtype=np.uint8)
        assert normalise(x).shape == x.shape


# ── Train / val split ─────────────────────────────────────────────────────────

class TestSplit:
    def test_sizes(self):
        x = np.zeros((1000, 32, 32, 3))
        y = np.zeros(1000, dtype=int)
        (x_t, y_t), (x_v, y_v) = split_train_val(x, y, val_split=0.2)
        assert len(x_t) == 800
        assert len(x_v) == 200

    def test_no_overlap(self):
        x = np.arange(1000)
        y = np.zeros(1000, dtype=int)
        (x_t, _), (x_v, _) = split_train_val(x.reshape(-1, 1, 1, 1), y)
        assert len(set(x_t.flatten()) & set(x_v.flatten())) == 0


# ── Baseline CNN ──────────────────────────────────────────────────────────────

class TestBaselineCNN:
    @pytest.fixture(scope="class")
    def model(self):
        return build_baseline_cnn()

    def test_output_shape(self, model):
        x = np.random.rand(4, *IMAGE_SIZE, 3).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (4, NUM_CLASSES)

    def test_probabilities_sum_to_one(self, model):
        x = np.random.rand(8, *IMAGE_SIZE, 3).astype(np.float32)
        out = model.predict(x, verbose=0)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(8), atol=1e-5)

    def test_model_name(self, model):
        assert model.name == "BaselineCNN"


# ── Transfer Learning Model ───────────────────────────────────────────────────

class TestTransferModel:
    @pytest.fixture(scope="class")
    def model(self):
        return build_transfer_model()

    def test_output_shape(self, model):
        x = np.random.rand(2, *IMAGE_SIZE_TL, 3).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (2, NUM_CLASSES)

    def test_backbone_frozen_by_default(self, model):
        base = model._base_model
        assert not base.trainable or all(not l.trainable for l in base.layers)

    def test_model_name(self, model):
        assert model.name == "TransferMobileNetV2"
