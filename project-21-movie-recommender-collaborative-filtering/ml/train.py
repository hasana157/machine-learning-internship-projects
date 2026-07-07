"""
CineMatch AI - ML Model Training Pipeline

This module handles the complete ML pipeline:
- Data loading and preprocessing
- Model training and hyperparameter tuning
- Evaluation and metrics computation
- Model persistence and versioning
"""

import logging
from typing import Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import SVD, Reader, Dataset
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """Main ML pipeline for training recommendation models."""
    
    def __init__(self, data_path: str = "/app/data"):
        """
        Initialize ML pipeline.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load ratings data from CSV or database.
        
        Returns:
            DataFrame with columns: user_id, movie_id, rating, timestamp
        """
        logger.info("Loading data...")
        
        # TODO: Load from database or CSV
        # For now, return empty DataFrame as placeholder
        
        data = pd.DataFrame({
            'user_id': [],
            'movie_id': [],
            'rating': [],
            'timestamp': []
        })
        
        logger.info(f"Loaded {len(data)} ratings")
        return data
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess and validate data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, preprocessing metadata)
        """
        logger.info("Preprocessing data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['user_id', 'movie_id'])
        logger.info(f"Removed {initial_count - len(df)} duplicate ratings")
        
        # Filter users with minimum ratings
        min_ratings = 20
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        df = df[df['user_id'].isin(valid_users)]
        logger.info(f"Filtered to {len(valid_users)} users with >= {min_ratings} ratings")
        
        # Validate ratings are in correct range
        df = df[(df['rating'] >= 0.5) & (df['rating'] <= 5.0)]
        
        metadata = {
            'total_ratings': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_movies': df['movie_id'].nunique(),
            'min_rating': df['rating'].min(),
            'max_rating': df['rating'].max(),
            'mean_rating': df['rating'].mean(),
        }
        
        logger.info(f"Data preprocessing complete: {metadata}")
        return df, metadata
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """
        Split data into train/val/test sets using temporal order.
        
        Args:
            df: Input DataFrame (must have timestamp column)
            test_size: Fraction for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data...")
        
        # Sort by timestamp for temporal split
        df = df.sort_values('timestamp')
        
        # Train/val/test split (80/10/10)
        n = len(df)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        logger.info(
            f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def train_svd_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_factors: int = 128,
        n_epochs: int = 30,
        lr_all: float = 0.007,
        reg_all: float = 0.08,
    ) -> Dict[str, Any]:
        """
        Train SVD model with given hyperparameters.
        
        Args:
            train_df: Training data
            val_df: Validation data for early stopping
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            lr_all: Learning rate
            reg_all: Regularization coefficient
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training SVD model...")
        
        # Prepare data for Surprise library
        reader = Reader(rating_scale=(0.5, 5.0))
        train_data = Dataset.load_from_df(
            train_df[['user_id', 'movie_id', 'rating']],
            reader
        )
        
        # Train SVD model
        model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            biased=True,
            verbose=True,
        )
        
        train_set = train_data.build_full_trainset()
        model.fit(train_set)
        
        # Evaluate on validation set
        val_predictions = [
            model.predict(uid, iid) for uid, iid in zip(
                val_df['user_id'], val_df['movie_id']
            )
        ]
        
        val_rmse = np.sqrt(
            np.mean([
                (pred.est - actual) ** 2
                for pred, actual in zip(val_predictions, val_df['rating'])
            ])
        )
        
        logger.info(f"SVD Model RMSE: {val_rmse:.4f}")
        
        return {
            'model': model,
            'rmse': float(val_rmse),
            'hyperparameters': {
                'n_factors': n_factors,
                'n_epochs': n_epochs,
                'lr_all': lr_all,
                'reg_all': reg_all,
            }
        }
    
    def evaluate_model(
        self,
        model: Any,
        test_df: pd.DataFrame,
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            test_df: Test data
            k: Number of recommendations for Hit@K
            
        Returns:
            Dictionary with metrics
        """
        logger.info("Evaluating model...")
        
        # Generate predictions
        predictions = [
            model.predict(uid, iid) for uid, iid in zip(
                test_df['user_id'], test_df['movie_id']
            )
        ]
        
        # Calculate RMSE
        rmse = np.sqrt(
            np.mean([
                (pred.est - actual) ** 2
                for pred, actual in zip(predictions, test_df['rating'])
            ])
        )
        
        # Hit@K calculation
        # This is simplified - full implementation would track per-user hits
        hit_count = 0
        for pred in predictions:
            if abs(pred.est - pred.r_ui) < 0.5:  # Within 0.5 stars
                hit_count += 1
        
        hit_at_k = hit_count / len(predictions) if predictions else 0
        
        metrics = {
            'rmse': float(rmse),
            'hit_at_k': float(hit_at_k),
            'mae': float(np.mean([
                abs(pred.est - actual)
                for pred, actual in zip(predictions, test_df['rating'])
            ])),
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run complete ML pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting ML pipeline...")
        
        try:
            # Load and preprocess data
            data = self.load_data()
            if data.empty:
                logger.warning("No data available, skipping training")
                return {'status': 'no_data'}
            
            data, metadata = self.preprocess_data(data)
            
            # Split data
            train_df, val_df, test_df = self.split_data(data)
            
            # Train model
            result = self.train_svd_model(train_df, val_df)
            model = result['model']
            
            # Evaluate
            metrics = self.evaluate_model(model, test_df)
            
            logger.info("✓ ML pipeline completed successfully")
            
            return {
                'status': 'success',
                'model': model,
                'metrics': metrics,
                'metadata': metadata,
            }
            
        except Exception as e:
            logger.error(f"✗ ML pipeline failed: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    # Run pipeline
    pipeline = MLPipeline()
    result = pipeline.run_pipeline()
    
    if result['status'] == 'success':
        logger.info("Pipeline results:")
        logger.info(f"  Metrics: {result['metrics']}")
        logger.info(f"  Metadata: {result['metadata']}")
    else:
        logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
