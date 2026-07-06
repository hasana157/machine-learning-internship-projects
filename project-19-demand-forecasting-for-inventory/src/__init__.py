"""
ForecastIQ: Production-Grade Demand Forecasting System for Retail Inventory

A complete MLOps pipeline for demand forecasting trained on the Rossmann Store Sales
Kaggle dataset. Includes data loading, feature engineering, model training, evaluation,
recursive forecasting, and an interactive Streamlit dashboard.

Modules:
    - data_loader: Load and clean Rossmann CSV data or generate synthetic fallback
    - features: Comprehensive feature engineering (28+ features)
    - model: DemandForecaster class (Random Forest + Linear Regression baseline)
    - trainer: Training pipeline with persistence
    - forecaster: Recursive future forecasting and scenario simulation
    - evaluator: Evaluation metrics, error analysis, report generation
    - utils: Config loading, logging, path helpers
"""

__version__ = "1.0.0"
__author__ = "Hasana Zahid"
__project__ = "ForecastIQ"
