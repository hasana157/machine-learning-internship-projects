"""
DemandForecaster model class wrapping Random Forest and Linear Regression.

Includes:
    - Preprocessing pipeline with categorical encoding and scaling
    - Random Forest main model
    - Linear Regression baseline
    - Model persistence (save/load)
    - Feature importance extraction
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import get_categorical_feature_cols, get_numeric_feature_cols

logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    Random Forest-based demand forecaster for retail sales.

    Attributes:
        pipeline: sklearn Pipeline with ColumnTransformer + RandomForestRegressor
        baseline_pipeline: sklearn Pipeline with ColumnTransformer + LinearRegression
        config: Model configuration dictionary
        trained_at: ISO timestamp of when model was last trained
        data_source: "kaggle" or "synthetic" indicating training data source
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize DemandForecaster with preprocessing and model pipelines.

        Args:
            config: Configuration dict with keys:
                - n_estimators: Number of RF trees
                - max_depth: Max tree depth
                - min_samples_leaf: Min samples per leaf
                - random_state: Random seed
        """
        num_cols = get_numeric_feature_cols()
        cat_cols = get_categorical_feature_cols()

        # Preprocessing: one-hot encode categoricals, scale numerics
        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols),
            ]
        )

        # Random Forest pipeline
        self.pipeline = Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=config.get("n_estimators", 150),
                        max_depth=config.get("max_depth", 20),
                        min_samples_leaf=config.get("min_samples_leaf", 2),
                        n_jobs=4,
                        random_state=config.get("random_state", 42),
                        oob_score=True,
                        verbose=1,
                    ),
                ),
            ]
        )

        # Baseline Linear Regression pipeline
        self.baseline_pipeline = Pipeline(
            [
                ("prep", preprocessor),
                ("model", LinearRegression()),
            ]
        )

        self.config = config
        self.trained_at: str | None = None
        self.data_source: str | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, data_source: str = "kaggle") -> None:
        """
        Train both RF and baseline models.

        Args:
            X_train: Training feature matrix.
            y_train: Training target (sales).
            data_source: "kaggle" or "synthetic" for metadata tracking.
        """
        logger.info("🌲 Training Random Forest model...")
        self.pipeline.fit(X_train, y_train)

        logger.info("📈 Training baseline Linear Regression...")
        self.baseline_pipeline.fit(X_train, y_train)

        self.trained_at = datetime.now().isoformat()
        self.data_source = data_source

        # Log OOB score
        oob_score = self.pipeline.named_steps["model"].oob_score_
        logger.info(f"✅ Training complete | OOB Score: {oob_score:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using Random Forest model.

        Args:
            X: Feature matrix (must have same columns as training data).

        Returns:
            Predicted sales values.
        """
        return self.pipeline.predict(X)

    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using Linear Regression baseline.

        Args:
            X: Feature matrix.

        Returns:
            Predicted sales values.
        """
        return self.baseline_pipeline.predict(X)

    def get_feature_importances(self) -> pd.Series:
        """
        Extract feature importances from trained Random Forest.

        Returns:
            Sorted Series of feature importances (descending).
        """
        rf_model = self.pipeline.named_steps["model"]
        importances = rf_model.feature_importances_

        # Get feature names from ColumnTransformer
        ct = self.pipeline.named_steps["prep"]
        feature_names = []

        # Categorical features (one-hot encoded)
        cat_encoder = ct.named_transformers_["cat"]
        if hasattr(cat_encoder, "get_feature_names_out"):
            cat_names = cat_encoder.get_feature_names_out(get_categorical_feature_cols())
            feature_names.extend(cat_names)

        # Numeric features
        numeric_names = get_numeric_feature_cols()
        feature_names.extend(numeric_names)

        # Create Series and sort
        importance_series = pd.Series(importances, index=feature_names)
        importance_series = importance_series.sort_values(ascending=False)

        return importance_series

    def save(self, path: str) -> None:
        """
        Save model and metadata to disk.

        Args:
            path: Path to save .joblib file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)
        logger.info(f"💾 Model saved to {path}")

        # Save metadata JSON
        metadata = {
            "trained_at": self.trained_at,
            "data_source": self.data_source,
            "oob_score": float(self.pipeline.named_steps["model"].oob_score_),
            "n_features": len(get_numeric_feature_cols()) + len(get_categorical_feature_cols()),
        }
        metadata_path = Path(path).parent / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"📋 Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, path: str) -> "DemandForecaster":
        """
        Load model from disk.

        Args:
            path: Path to .joblib file.

        Returns:
            Loaded DemandForecaster instance.

        Raises:
            FileNotFoundError: If model file not found.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model = joblib.load(path)
        logger.info(f"✅ Model loaded from {path}")
        return model
