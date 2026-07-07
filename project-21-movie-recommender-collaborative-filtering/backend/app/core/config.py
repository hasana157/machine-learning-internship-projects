"""
Configuration management for CineMatch AI backend.

Uses Pydantic Settings for environment-based configuration.
Supports development, staging, and production environments.
"""

from typing import List
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Follows 12-factor app methodology with environment-based configuration.
    """
    
    # ============================================================================
    # ENVIRONMENT & GENERAL
    # ============================================================================
    
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    
    API_TITLE: str = Field(default="CineMatch AI", description="API title")
    API_VERSION: str = Field(default="1.0.0", description="API version")
    API_DESCRIPTION: str = Field(default="Collaborative Filtering Movie Recommendation Engine")
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_WORKERS: int = Field(default=4, description="Number of API workers")
    
    # ============================================================================
    # DATABASE
    # ============================================================================
    
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/cinematch_db",
        description="PostgreSQL connection URL",
    )
    DB_POOL_SIZE: int = Field(default=20, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=10, description="Max overflow connections")
    DB_POOL_RECYCLE: int = Field(default=3600, description="Pool recycle time in seconds")
    DB_ECHO: bool = Field(default=False, description="Echo SQL queries")
    
    # ============================================================================
    # REDIS/CACHE
    # ============================================================================
    
    REDIS_URL: str = Field(
        default="redis://:password@localhost:6379/0",
        description="Redis connection URL",
    )
    REDIS_CACHE_TTL: int = Field(default=3600, description="Default cache TTL in seconds")
    REDIS_POOL_SIZE: int = Field(default=10, description="Redis connection pool size")
    
    # ============================================================================
    # SECURITY
    # ============================================================================
    
    SECRET_KEY: str = Field(
        default="change-this-in-production",
        description="Secret key for signing tokens",
    )
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration in minutes")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration in days")
    
    # ============================================================================
    # CORS
    # ============================================================================
    
    ALLOWED_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8000",
        ],
        description="CORS allowed origins",
    )
    ALLOWED_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="CORS allowed methods",
    )
    ALLOWED_HEADERS: List[str] = Field(
        default=["*"],
        description="CORS allowed headers",
    )
    
    # ============================================================================
    # RATE LIMITING
    # ============================================================================
    
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, description="Requests per minute limit")
    
    # ============================================================================
    # ML/AI CONFIGURATION
    # ============================================================================
    
    ML_MODEL_NAME: str = Field(default="svd_ensemble_v1", description="ML model name")
    ML_DEFAULT_K: int = Field(default=10, description="Default number of recommendations")
    ML_MIN_USER_RATINGS: int = Field(default=5, description="Minimum ratings for cold-start handling")
    ML_CACHE_MODEL: bool = Field(default=True, description="Cache loaded models")
    ML_MODEL_UPDATE_INTERVAL: int = Field(default=3600, description="Model update interval in seconds")
    
    # SVD Hyperparameters
    SVD_N_FACTORS: int = Field(default=128, description="Number of latent factors")
    SVD_N_EPOCHS: int = Field(default=30, description="Number of training epochs")
    SVD_LR_ALL: float = Field(default=0.007, description="Learning rate")
    SVD_REG_ALL: float = Field(default=0.08, description="Regularization coefficient")
    SVD_BIASED: bool = Field(default=True, description="Use biased SVD")
    
    # Evaluation Metrics
    EVAL_HIT_TARGET: float = Field(default=0.65, description="Target Hit@10")
    EVAL_NDCG_TARGET: float = Field(default=0.42, description="Target NDCG@10")
    EVAL_COVERAGE_TARGET: float = Field(default=0.40, description="Target coverage")
    
    # ============================================================================
    # MONITORING & OBSERVABILITY
    # ============================================================================
    
    PROMETHEUS_ENABLED: bool = Field(default=True, description="Enable Prometheus metrics")
    PROMETHEUS_PORT: int = Field(default=9090, description="Prometheus port")
    
    MLFLOW_TRACKING_URI: str = Field(
        default="http://mlflow:5000",
        description="MLflow tracking server URI",
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(default="cinematch-dev")
    
    # ============================================================================
    # AWS CONFIGURATION
    # ============================================================================
    
    AWS_REGION: str = Field(default="us-east-1", description="AWS region")
    AWS_S3_BUCKET: str = Field(default="cinematch-models", description="S3 bucket for models")
    AWS_ACCESS_KEY_ID: str = Field(default="", description="AWS access key")
    AWS_SECRET_ACCESS_KEY: str = Field(default="", description="AWS secret key")
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    
    FEATURE_ENABLE_ENSEMBLE: bool = Field(default=True, description="Enable ensemble recommendations")
    FEATURE_ENABLE_IMPLICIT_FEEDBACK: bool = Field(default=True, description="Enable implicit feedback")
    FEATURE_ENABLE_HYBRID: bool = Field(default=False, description="Enable hybrid recommendations")
    FEATURE_ENABLE_SOCIAL: bool = Field(default=False, description="Enable social features")
    FEATURE_ENABLE_A_B_TESTING: bool = Field(default=True, description="Enable A/B testing")
    
    # ============================================================================
    # CELERY (Background Tasks)
    # ============================================================================
    
    CELERY_BROKER_URL: str = Field(
        default="redis://:password@localhost:6379/1",
        description="Celery broker URL",
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://:password@localhost:6379/2",
        description="Celery result backend URL",
    )
    CELERY_TASK_ALWAYS_EAGER: bool = Field(default=True, description="Execute tasks synchronously")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v: str, values: dict) -> str:
        """Ensure secret key is secure in production"""
        if values.get("ENVIRONMENT") == "production":
            if len(v) < 32 or v == "change-this-in-production":
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters long in production"
                )
        return v
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to avoid re-parsing environment variables on each access.
    
    Returns:
        Settings: Application configuration instance
    """
    return Settings()


# Create global settings instance
settings = get_settings()
