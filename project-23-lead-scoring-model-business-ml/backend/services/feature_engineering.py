"""
Feature Engineering Service
Transforms raw CRM data into ML features
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature Engineering Pipeline
    Transforms raw CRM lead data into ML-ready features
    """
    
    # Feature configuration
    NUMERIC_FEATURES = [
        'company_size',
        'email_opens',
        'email_clicks',
        'days_since_contact',
        'deal_value',
        'engagement_score'
    ]
    
    CATEGORICAL_FEATURES = [
        'industry',
        'last_activity_type'
    ]
    
    # Default values for missing features
    DEFAULTS = {
        'company_size': 100,
        'email_opens': 0,
        'email_clicks': 0,
        'days_since_contact': 30,
        'deal_value': 0.0,
        'engagement_score': 0.0,
        'industry': 'unknown',
        'last_activity_type': 'email'
    }
    
    # Industry mappings
    INDUSTRY_MAP = {
        'saas': 0,
        'finance': 1,
        'healthcare': 2,
        'retail': 3,
        'manufacturing': 4,
        'technology': 5,
        'unknown': 6
    }
    
    # Activity type mappings
    ACTIVITY_MAP = {
        'email': 0,
        'call': 1,
        'meeting': 2,
        'demo': 3,
        'proposal': 4,
        'unknown': 5
    }
    
    def __init__(self):
        """Initialize Feature Engineer"""
        self.feature_count = len(self.NUMERIC_FEATURES) + len(self.CATEGORICAL_FEATURES) + 5
        logger.info(f"Initialized FeatureEngineer with {self.feature_count} features")
    
    def transform_single(self, data: Dict) -> np.ndarray:
        """
        Transform a single lead record into ML features
        
        Args:
            data: Dictionary with lead features
        
        Returns:
            Feature array (1D numpy array)
        """
        try:
            features = []
            
            # Process numeric features
            for feature in self.NUMERIC_FEATURES:
                value = data.get(feature, self.DEFAULTS.get(feature, 0))
                features.append(self._normalize_numeric(feature, value))
            
            # Process categorical features
            for feature in self.CATEGORICAL_FEATURES:
                value = data.get(feature, self.DEFAULTS.get(feature, 'unknown'))
                features.append(self._encode_categorical(feature, value))
            
            # Engineered features
            engineered = self._create_engineered_features(data)
            features.extend(engineered)
            
            # Ensure correct shape
            features_array = np.array(features, dtype=np.float32)
            
            if features_array.shape[0] != self.feature_count:
                logger.warning(
                    f"Feature count mismatch: expected {self.feature_count}, "
                    f"got {features_array.shape[0]}"
                )
                # Pad with zeros if too short
                if features_array.shape[0] < self.feature_count:
                    padding = np.zeros(self.feature_count - features_array.shape[0])
                    features_array = np.concatenate([features_array, padding])
                else:
                    # Truncate if too long
                    features_array = features_array[:self.feature_count]
            
            return features_array.reshape(1, -1)
        
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            # Return zeros if transformation fails
            return np.zeros((1, self.feature_count), dtype=np.float32)
    
    def transform_batch(self, data_list: List[Dict]) -> np.ndarray:
        """
        Transform multiple lead records
        
        Args:
            data_list: List of lead dictionaries
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        try:
            features_list = []
            for data in data_list:
                features = self.transform_single(data).flatten()
                features_list.append(features)
            
            return np.array(features_list, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error in batch transformation: {e}")
            return np.zeros((len(data_list), self.feature_count), dtype=np.float32)
    
    def _normalize_numeric(self, feature_name: str, value) -> float:
        """
        Normalize numeric feature
        
        Args:
            feature_name: Name of the feature
            value: Raw value
        
        Returns:
            Normalized value
        """
        try:
            # Convert to float, handle None/missing
            if value is None or value == '':
                value = self.DEFAULTS.get(feature_name, 0)
            else:
                value = float(value)
            
            # Feature-specific normalization
            if feature_name == 'company_size':
                # Log scale normalization (1-100000)
                return np.log1p(value) / np.log1p(100000)
            
            elif feature_name == 'email_opens':
                # Clip and normalize (0-1000)
                return min(float(value), 1000) / 1000.0
            
            elif feature_name == 'email_clicks':
                # Clip and normalize (0-1000)
                return min(float(value), 1000) / 1000.0
            
            elif feature_name == 'days_since_contact':
                # Recency: recent is better (invert after normalization)
                return 1.0 - (min(float(value), 365) / 365.0)
            
            elif feature_name == 'deal_value':
                # Log scale normalization
                return np.log1p(value) / np.log1p(10000000)
            
            elif feature_name == 'engagement_score':
                # Already 0-100 normalized
                return float(value) / 100.0
            
            else:
                return float(value)
        
        except Exception as e:
            logger.warning(f"Error normalizing {feature_name}: {e}")
            return 0.0
    
    def _encode_categorical(self, feature_name: str, value: str) -> int:
        """
        Encode categorical feature
        
        Args:
            feature_name: Name of the feature
            value: Raw value
        
        Returns:
            Encoded value
        """
        try:
            if value is None or value == '':
                value = self.DEFAULTS.get(feature_name, 'unknown')
            
            value = str(value).lower().strip()
            
            if feature_name == 'industry':
                return float(self.INDUSTRY_MAP.get(value, self.INDUSTRY_MAP['unknown']))
            
            elif feature_name == 'last_activity_type':
                return float(self.ACTIVITY_MAP.get(value, self.ACTIVITY_MAP['unknown']))
            
            else:
                return 0.0
        
        except Exception as e:
            logger.warning(f"Error encoding {feature_name}: {e}")
            return 0.0
    
    def _create_engineered_features(self, data: Dict) -> List[float]:
        """
        Create derived features from raw data
        
        Args:
            data: Raw lead features
        
        Returns:
            List of engineered features
        """
        features = []
        
        try:
            # Feature 1: Email Engagement Score
            opens = data.get('email_opens', 0) or 0
            clicks = data.get('email_clicks', 0) or 0
            email_engagement = (opens + clicks * 2) / 100.0  # Clicks weighted 2x
            features.append(min(email_engagement, 1.0))
            
            # Feature 2: Deal Value per Company Size (deal size relative to company)
            deal_value = data.get('deal_value', 0) or 0
            company_size = data.get('company_size', 100) or 100
            deal_per_size = (deal_value / max(company_size, 1)) / 10000.0
            features.append(min(deal_per_size, 1.0))
            
            # Feature 3: Recency-Based Score (recent activity is good)
            days_since = data.get('days_since_contact', 30) or 30
            recency_score = np.exp(-days_since / 30)  # Exponential decay
            features.append(float(recency_score))
            
            # Feature 4: Engagement Level (normalized engagement_score)
            engagement = data.get('engagement_score', 0) or 0
            features.append(float(engagement) / 100.0)
            
            # Feature 5: Activity Frequency (if we had multiple activity types)
            # Placeholder for future expansion
            features.append(0.5)
            
            return features
        
        except Exception as e:
            logger.warning(f"Error creating engineered features: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names in order
        
        Returns:
            List of feature names
        """
        names = (
            self.NUMERIC_FEATURES +
            self.CATEGORICAL_FEATURES +
            [
                'email_engagement_score',
                'deal_per_company_size',
                'recency_score',
                'engagement_level',
                'activity_frequency'
            ]
        )
        return names
    
    @staticmethod
    def create_dataframe(data_list: List[Dict]) -> pd.DataFrame:
        """
        Create pandas DataFrame from lead records (for analysis/training)
        
        Args:
            data_list: List of lead dictionaries
        
        Returns:
            pandas DataFrame
        """
        try:
            df = pd.DataFrame(data_list)
            
            # Fill missing numeric values with defaults
            numeric_cols = [
                'company_size', 'email_opens', 'email_clicks',
                'days_since_contact', 'deal_value', 'engagement_score'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # Fill missing categorical values with 'unknown'
            categorical_cols = ['industry', 'last_activity_type']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('unknown')
            
            return df
        
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
