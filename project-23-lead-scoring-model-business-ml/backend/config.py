"""
Configuration Management
Settings for API, database, and ML components
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application Settings"""
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_VERSION: str = "1.0.0"
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "your-secret-key-change-in-prod")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://leadforge:password@localhost:5432/leadforge_db"
    )
    SQLALCHEMY_ECHO: bool = bool(os.getenv("SQLALCHEMY_ECHO", False))
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_TIMEOUT: int = int(os.getenv("REDIS_TIMEOUT", 300))
    
    # Model Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/xgboost_model.pkl")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "models/scaler.pkl")
    CALIBRATOR_PATH: str = os.getenv("CALIBRATOR_PATH", "models/calibrator.pkl")
    FEATURE_NAMES_PATH: str = os.getenv("FEATURE_NAMES_PATH", "models/feature_names.json")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8050",
        "http://localhost:8000"
    ]
    
    if os.getenv("ALLOWED_ORIGINS"):
        ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS").split(",")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Feature Engineering
    FEATURE_SCALING: bool = True
    HANDLE_MISSING_VALUES: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
