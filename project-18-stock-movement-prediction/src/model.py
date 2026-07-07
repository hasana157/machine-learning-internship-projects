"""
ML models for stock movement prediction.
"""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter

logger = logging.getLogger(__name__)


class StockMovementPredictor:
    """Stock movement prediction model using Random Forest and Logistic Regression."""
    
    def __init__(self, config: dict):
        """Initialize the predictor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Random Forest
        rf_params = config['model']['random_forest']
        self.rf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params['min_samples_leaf'],
            max_features=rf_params['max_features'],
            random_state=rf_params['random_state'],
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True
        )
        
        # Logistic Regression baseline
        lr_params = config['model']['logistic_regression']
        self.lr_baseline = LogisticRegression(
            max_iter=lr_params['max_iter'],
            random_state=lr_params['random_state'],
            class_weight='balanced',
            C=lr_params['C']
        )
        
        # Scaler
        self.scaler = StandardScaler()
        
        # Metadata
        self.feature_names = None
        self.training_date_range = None
        self.ticker = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, ticker: str = None) -> None:
        """Train both models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            ticker: Stock ticker symbol
        """
        self.ticker = ticker
        self.training_date_range = (X.index.min(), X.index.max())
        
        # Validate
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Random Forest
        self.rf_model.fit(X_scaled, y)
        
        # Fit Logistic Regression
        self.lr_baseline.fit(X_scaled, y)
        
        self.is_fitted = True
        
        # Log info
        logger.info(f"✅ Trained on {len(X)} samples, {len(self.feature_names)} features")
        logger.info(f"   Class distribution: {dict(Counter(y))}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using Random Forest.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from Random Forest.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability array, shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict_proba(X_scaled)
    
    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using Logistic Regression baseline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        return self.lr_baseline.predict(X_scaled)
    
    def predict_proba_baseline(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from Logistic Regression.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability array, shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        return self.lr_baseline.predict_proba(X_scaled)
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        threshold: float = 0.55
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence score.
        
        Args:
            X: Feature DataFrame
            threshold: Confidence threshold (default 0.55)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        proba = self.predict_proba(X)[:, 1]  # Probability of UP (class 1)
        predictions = (proba >= threshold).astype(int)
        
        # Confidence score: distance from 0.5
        confidence = np.abs(proba - 0.5) * 2
        
        return predictions, confidence
    
    def get_feature_importances(self, top_n: int = 20) -> pd.Series:
        """Get top N feature importances from Random Forest.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Series with feature names and importances
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        importances = self.rf_model.feature_importances_
        feature_importance = pd.Series(
            importances,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return feature_importance.head(top_n)
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Metadata dictionary
        """
        return {
            'ticker': self.ticker,
            'training_date_range': {
                'start': str(self.training_date_range[0]) if self.training_date_range else None,
                'end': str(self.training_date_range[1]) if self.training_date_range else None
            },
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'model_type': 'RandomForest+LogisticRegression',
            'is_fitted': self.is_fitted
        }
    
    def save(self, path: str) -> None:
        """Save model to disk using joblib.
        
        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'StockMovementPredictor':
        """Load model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded model
        """
        model = joblib.load(path)
        logger.info(f"✅ Model loaded from {path}")
        return model
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """Validate that X has same features as training data.
        
        Args:
            X: DataFrame to validate
            
        Raises:
            ValueError if features don't match
        """
        if self.feature_names is None:
            raise ValueError("Model not fitted")
        
        if set(X.columns) != set(self.feature_names):
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            if missing:
                logger.warning(f"⚠️  Missing features: {missing}")
            if extra:
                logger.warning(f"⚠️  Extra features: {extra}")
            raise ValueError("Feature mismatch")
