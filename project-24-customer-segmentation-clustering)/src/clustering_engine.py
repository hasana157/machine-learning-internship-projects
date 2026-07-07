"""
clustering_engine.py
---------------------
KMeans clustering with silhouette-driven selection of the optimal number
of clusters (k), plus supplementary validation metrics.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)


@dataclass
class ClusteringResult:
    best_k: int
    silhouette_score: float
    davies_bouldin: float
    calinski_harabasz: float
    inertia: float
    k_search: list = field(default_factory=list)  # list of {k, silhouette, inertia}


class ClusteringEngine:
    """Fit KMeans across a range of k, pick the best by silhouette score,
    and expose validation metrics + inverse-scaled centroids.
    """

    def __init__(self, k_range=range(3, 6), random_state: int = 42,
                 n_init: int = 10, max_iter: int = 300,
                 min_silhouette: float = 0.30, plateau_tolerance: float = 0.02):
        self.k_range = list(k_range)
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.min_silhouette = min_silhouette
        self.plateau_tolerance = plateau_tolerance

        self.model: Optional[KMeans] = None
        self.best_k: Optional[int] = None
        self.result: Optional[ClusteringResult] = None
        self._silhouette_samples_: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, rfm_scaled: np.ndarray) -> "ClusteringEngine":
        """Grid-search k, refit the winner with a larger n_init, and store
        all validation metrics.
        """
        rfm_scaled = np.asarray(rfm_scaled)
        if rfm_scaled.ndim != 2 or rfm_scaled.shape[1] < 2:
            raise ValueError("rfm_scaled must be a 2D array with >= 2 feature columns.")
        if len(rfm_scaled) < max(self.k_range) + 1:
            raise ValueError("Not enough samples for the requested k range.")

        k_search = []
        for k in self.k_range:
            km = KMeans(
                n_clusters=k, init="k-means++", max_iter=self.max_iter,
                random_state=self.random_state, n_init=self.n_init,
            )
            labels = km.fit_predict(rfm_scaled)
            score = silhouette_score(rfm_scaled, labels)
            k_search.append({"k": k, "silhouette": float(score), "inertia": float(km.inertia_)})

        # Primary: argmax silhouette. Secondary: prefer lower k on a plateau.
        best_score = max(r["silhouette"] for r in k_search)
        candidates = [r["k"] for r in k_search if best_score - r["silhouette"] <= self.plateau_tolerance]
        best_k = min(candidates)

        if best_score < self.min_silhouette:
            import warnings
            warnings.warn(
                f"Best silhouette score ({best_score:.3f}) is below the "
                f"recommended minimum ({self.min_silhouette}). Clusters may overlap; "
                f"consider different features or more data."
            )

        # Refit winner with a higher n_init for robustness
        final_km = KMeans(
            n_clusters=best_k, init="k-means++", max_iter=self.max_iter,
            random_state=self.random_state, n_init=max(self.n_init, 20),
        )
        labels = final_km.fit_predict(rfm_scaled)

        self.model = final_km
        self.best_k = best_k
        self._labels = labels
        self._silhouette_samples_ = silhouette_samples(rfm_scaled, labels)

        self.result = ClusteringResult(
            best_k=best_k,
            silhouette_score=float(silhouette_score(rfm_scaled, labels)),
            davies_bouldin=float(davies_bouldin_score(rfm_scaled, labels)),
            calinski_harabasz=float(calinski_harabasz_score(rfm_scaled, labels)),
            inertia=float(final_km.inertia_),
            k_search=k_search,
        )
        return self

    def predict(self, rfm_scaled: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(np.asarray(rfm_scaled))

    def get_centroids_original(self, rfm_calculator) -> pd.DataFrame:
        """Inverse-scale cluster centroids back into original R/F/M units.

        `rfm_calculator` must be a fitted RFMCalculator (or any object
        exposing `inverse_transform`).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before get_centroids_original().")
        original = rfm_calculator.inverse_transform(self.model.cluster_centers_)
        return pd.DataFrame(original, columns=["Recency", "Frequency", "Monetary"])

    def get_silhouette_samples(self) -> np.ndarray:
        if self._silhouette_samples_ is None:
            raise RuntimeError("Call fit() before get_silhouette_samples().")
        return self._silhouette_samples_

    def get_labels(self) -> np.ndarray:
        if self._labels is None:
            raise RuntimeError("Call fit() before get_labels().")
        return self._labels

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ClusteringEngine":
        with open(path, "rb") as f:
            return pickle.load(f)
