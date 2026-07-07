"""
Database configuration and initialization.

Uses SQLAlchemy 2.x with async support via asyncpg driver.
Provides connection pool management and session factory.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from app.core.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy ORM base
Base = declarative_base()

# Global engine and session factory
engine = None
AsyncSessionLocal = None


async def init_db() -> None:
    """
    Initialize database connection pool and create tables.
    
    Should be called once during application startup.
    """
    global engine, AsyncSessionLocal
    
    try:
        # Create async engine with connection pooling
        engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_recycle=settings.DB_POOL_RECYCLE,
            pool_pre_ping=True,
            echo=settings.DB_ECHO,
            future=True,
        )
        
        # Create session factory
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        
        # Test connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        logger.info("✓ Database connection pool initialized successfully")
        
        # Create tables (in production, use Alembic migrations)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✓ Database tables created/verified")
        
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise


async def close_db() -> None:
    """
    Close database connections.
    
    Should be called during application shutdown.
    """
    global engine
    
    if engine is not None:
        await engine.dispose()
        logger.info("✓ Database connections closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session.
    
    Yields:
        AsyncSession: Database session for query execution
    """
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# ORM MODELS
# ============================================================================


from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Index,
    JSON,
    Text,
    DECIMAL,
)
from datetime import datetime
from sqlalchemy.orm import relationship


class User(Base):
    """User account model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    events = relationship("UserEvent", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class Movie(Base):
    """Movie catalog model"""
    __tablename__ = "movies"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    release_year = Column(Integer, nullable=True, index=True)
    tmdb_id = Column(Integer, unique=True, nullable=True, index=True)
    imdb_id = Column(String(20), unique=True, nullable=True, index=True)
    poster_url = Column(String(512), nullable=True)
    overview = Column(Text, nullable=True)
    runtime = Column(Integer, nullable=True)
    rating = Column(Float, nullable=True)  # Average rating
    popularity = Column(Float, default=0.0, index=True)
    genres = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ratings = relationship("Rating", back_populates="movie", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_movie_title_year", "title", "release_year"),
    )
    
    def __repr__(self):
        return f"<Movie(id={self.id}, title={self.title}, year={self.release_year})>"


class Rating(Base):
    """User-movie ratings model"""
    __tablename__ = "ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)  # 0.5 to 5.0
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")
    
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie"),
        Index("idx_rating_user_movie", "user_id", "movie_id"),
        Index("idx_rating_timestamp", "timestamp"),
    )
    
    def __repr__(self):
        return f"<Rating(user_id={self.user_id}, movie_id={self.movie_id}, rating={self.rating})>"


class UserEvent(Base):
    """User interaction events (clicks, watches, views)"""
    __tablename__ = "user_events"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # click, view, watch, add_to_list
    duration_seconds = Column(Integer, nullable=True)
    metadata = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="events")
    
    __table_args__ = (
        Index("idx_event_user_type", "user_id", "event_type"),
        Index("idx_event_timestamp", "timestamp"),
    )
    
    def __repr__(self):
        return f"<UserEvent(user_id={self.user_id}, event_type={self.event_type})>"


class ModelRun(Base):
    """Model training runs and versions"""
    __tablename__ = "model_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    hit_at_10 = Column(Float, nullable=False)
    ndcg_at_10 = Column(Float, nullable=False)
    coverage = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    training_duration_seconds = Column(Integer, nullable=False)
    training_samples = Column(Integer, nullable=False)
    hyperparameters = Column(JSON, default=dict)
    model_path = Column(String(512), nullable=False)
    is_production = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_model_version", "model_name", "model_version"),
    )
    
    def __repr__(self):
        return f"<ModelRun(id={self.id}, model={self.model_name}, v={self.model_version})>"


class RecommendationCache(Base):
    """Pre-computed recommendations cache"""
    __tablename__ = "recommendation_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    movie_ids = Column(JSON, nullable=False)  # List of movie IDs
    scores = Column(JSON, nullable=False)  # List of prediction scores
    strategy = Column(String(50), default="ensemble")
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    
    def __repr__(self):
        return f"<RecommendationCache(user_id={self.user_id})>"


# Create indexes for better query performance
Index("idx_user_event_timestamp_type", UserEvent.timestamp, UserEvent.event_type)
Index("idx_movie_popularity", Movie.popularity)
Index("idx_rating_user_timestamp", Rating.user_id, Rating.timestamp)
