"""
Model module for SentinelFlow.
Contains the AnomalyDetector class wrapping IsolationForest and Z-Score logic.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

from src.utils import setup_logger

logger = setup_logger(__name__)

class AnomalyDetector:
    """
    Anomaly Detection model using Isolation Forest and Z-Score baseline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        model_cfg = config.get("model", {})
        self.n_estimators = model_cfg.get("n_estimators", 200)
        self.contamination = model_cfg.get("contamination", 0.03)
        self.random_state = model_cfg.get("random_state", 42)
        self.zscore_threshold = model_cfg.get("zscore_threshold", 3.0)
        
        self.iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
        self.threshold = 0.0
        self.feature_names: List[str] = []
        self._is_fitted = False
        self._training_features: pd.DataFrame = None

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the anomaly detector to normal data.

        Args:
            X (pd.DataFrame): Training features (ideally normal data).
        """
        logger.info(f"Fitting AnomalyDetector on {len(X)} samples...")
        self.feature_names = list(X.columns)
        
        # Keep a copy for feature importance calculation
        self._training_features = X.copy()
        
        # Scale and fit
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        
        # Calculate decision threshold (95th percentile of anomaly scores on training set)
        scores = -self.iso_forest.score_samples(X_scaled) # Negative because lower is more anomalous
        self.threshold = np.percentile(scores, 95)
        
        self._is_fitted = True
        logger.info(f"Model fitted. Threshold set to {self.threshold:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using Isolation Forest.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Binary array (1 = anomaly, 0 = normal).
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        # Ensure column order
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # IsolationForest returns 1 for inliers, -1 for outliers
        preds = self.iso_forest.predict(X_scaled)
        
        # Remap to 0 for normal, 1 for anomaly
        return np.where(preds == -1, 1, 0)

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized anomaly scores.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Normalized anomaly scores (0 to 1 range, 1 is most anomalous).
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Raw scores from IF (negative values, lower is more anomalous)
        raw_scores = self.iso_forest.score_samples(X_scaled)
        
        # Invert so higher is more anomalous
        inverted_scores = -raw_scores
        
        # Normalize to 0-1 (approximate based on IF bounds -0.5 to 0.5)
        # Using min-max normalization based on a reasonable range for IF scores
        min_score = 0.3  # Approx minimum inverted score for normal points
        max_score = 0.8  # Approx maximum inverted score for anomalous points
        
        norm_scores = (inverted_scores - min_score) / (max_score - min_score)
        return np.clip(norm_scores, 0, 1)

    def predict_zscore(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using Z-Score baseline across primary sensors.

        Args:
            X (pd.DataFrame): Input features containing z_score columns.

        Returns:
            np.ndarray: Binary array (1 = anomaly, 0 = normal).
        """
        z_cols = [c for c in X.columns if c.startswith("z_")]
        if not z_cols:
            logger.warning("No Z-score columns found. Returning all zeros for Z-Score prediction.")
            return np.zeros(len(X), dtype=int)
            
        # Get maximum absolute z-score across all sensor columns for each row
        max_z = X[z_cols].abs().max(axis=1).values
        
        return (max_z > self.zscore_threshold).astype(int)

    def get_feature_importances(self) -> pd.Series:
        """
        Compute permutation feature importances.
        Calculates drop in average anomaly score when a feature is shuffled.

        Returns:
            pd.Series: Feature importances sorted in descending order.
        """
        if not self._is_fitted or self._training_features is None:
            raise ValueError("Model must be fitted to compute feature importances.")
            
        logger.info("Computing feature importances...")
        
        baseline_scores = self.score_samples(self._training_features)
        baseline_mean = np.mean(baseline_scores)
        
        importances = {}
        for col in self.feature_names:
            X_shuffled = self._training_features.copy()
            # Shuffle the column
            X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
            
            # Recalculate scores
            shuffled_scores = self.score_samples(X_shuffled)
            
            # The importance is the absolute difference in mean anomaly score
            # (how much the model relied on this feature being ordered correctly)
            diff = np.abs(np.mean(shuffled_scores) - baseline_mean)
            importances[col] = diff
            
        # Normalize to sum to 1
        total_importance = sum(importances.values()) + 1e-9
        for col in importances:
            importances[col] /= total_importance
            
        return pd.Series(importances).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path (str): File path to save the model.
        """
        state = {
            "iso_forest": self.iso_forest,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "zscore_threshold": self.zscore_threshold,
            "is_fitted": self._is_fitted,
            "training_features": self._training_features
        }
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """
        Load the model from disk.

        Args:
            path (str): File path to load the model from.

        Returns:
            AnomalyDetector: Loaded model instance.
        """
        # Create a dummy config to initialize
        dummy_config = {"model": {}}
        instance = cls(dummy_config)
        
        state = joblib.load(path)
        instance.iso_forest = state["iso_forest"]
        instance.scaler = state["scaler"]
        instance.threshold = state["threshold"]
        instance.feature_names = state["feature_names"]
        instance.zscore_threshold = state["zscore_threshold"]
        instance._is_fitted = state["is_fitted"]
        instance._training_features = state.get("training_features", None)
        
        logger.info(f"Model loaded from {path}")
        return instance
