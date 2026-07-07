import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clustering_engine import ClusteringEngine


@pytest.fixture
def blob_data():
    """Three well-separated Gaussian blobs -> should cluster cleanly."""
    rng = np.random.default_rng(1)
    centers = np.array([[5, 5, 5], [-5, -5, -5], [5, -5, 0]])
    points = []
    for c in centers:
        points.append(rng.normal(loc=c, scale=0.5, size=(60, 3)))
    return np.vstack(points)


def test_fit_selects_reasonable_k(blob_data):
    engine = ClusteringEngine(k_range=range(2, 6))
    engine.fit(blob_data)
    assert engine.best_k in range(2, 6)
    assert engine.result.silhouette_score > 0.5


def test_predict_after_fit(blob_data):
    engine = ClusteringEngine(k_range=range(2, 6))
    engine.fit(blob_data)
    labels = engine.predict(blob_data[:5])
    assert len(labels) == 5


def test_predict_before_fit_raises(blob_data):
    engine = ClusteringEngine()
    with pytest.raises(RuntimeError):
        engine.predict(blob_data)


def test_get_labels_and_silhouette_samples(blob_data):
    engine = ClusteringEngine(k_range=range(2, 6))
    engine.fit(blob_data)
    labels = engine.get_labels()
    samples = engine.get_silhouette_samples()
    assert len(labels) == len(blob_data)
    assert len(samples) == len(blob_data)


def test_result_metrics_present(blob_data):
    engine = ClusteringEngine(k_range=range(2, 6))
    engine.fit(blob_data)
    r = engine.result
    assert r.silhouette_score is not None
    assert r.davies_bouldin >= 0
    assert r.calinski_harabasz > 0
    assert r.inertia > 0
    assert len(r.k_search) == 4


def test_invalid_input_shape_raises():
    engine = ClusteringEngine(k_range=range(2, 4))
    with pytest.raises(ValueError):
        engine.fit(np.array([1, 2, 3]))  # 1D array


def test_too_few_samples_raises():
    engine = ClusteringEngine(k_range=range(2, 11))
    with pytest.raises(ValueError):
        engine.fit(np.random.rand(5, 3))  # fewer samples than max(k_range)+1


def test_save_and_load_roundtrip(tmp_path, blob_data):
    engine = ClusteringEngine(k_range=range(2, 6))
    engine.fit(blob_data)
    path = tmp_path / "engine.pkl"
    engine.save(path)
    loaded = ClusteringEngine.load(path)
    np.testing.assert_array_equal(loaded.get_labels(), engine.get_labels())


def test_low_silhouette_warns():
    rng = np.random.default_rng(2)
    # Uniform noise -> no real cluster structure -> low silhouette expected
    data = rng.uniform(-1, 1, size=(120, 3))
    engine = ClusteringEngine(k_range=range(3, 6), min_silhouette=0.9)
    with pytest.warns(UserWarning):
        engine.fit(data)
