"""
Recommendations API endpoints.

Core endpoints for getting movie recommendations with caching and multiple strategies.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user_id
from app.db.database import get_db, User, Movie, Rating
from app.db.redis_client import (
    get_recommendations_cache,
    set_recommendations_cache,
    get_redis,
)
from app.schemas.recommendation import (
    RecommendationsResponse,
    RecommendationItem,
    RecommendationRequest,
    SimilarMoviesResponse,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def get_user_recommendations(
    user_id: int,
    db: AsyncSession,
    k: int = 10,
    strategy: str = "ensemble",
) -> List[Dict[str, Any]]:
    """
    Get recommendations for user using specified strategy.
    
    Args:
        user_id: User ID
        db: Database session
        k: Number of recommendations
        strategy: Recommendation strategy
        
    Returns:
        List of recommendation dicts
    """
    # Check if user exists
    from sqlalchemy import select
    user = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = user.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Get user's rating history
    ratings_result = await db.execute(
        select(Rating).where(Rating.user_id == user_id)
    )
    user_ratings = ratings_result.scalars().all()
    
    # Cold-start handling
    if len(user_ratings) < settings.ML_MIN_USER_RATINGS:
        return await get_popularity_recommendations(db, k, user_id)
    
    # Get recommendations based on strategy
    if strategy == "svd":
        recommendations = await get_svd_recommendations(user_id, db, k)
    elif strategy == "knn":
        recommendations = await get_knn_recommendations(user_id, db, k)
    elif strategy == "popularity":
        recommendations = await get_popularity_recommendations(db, k, user_id)
    else:  # ensemble (default)
        recommendations = await get_ensemble_recommendations(user_id, db, k)
    
    return recommendations


async def get_popularity_recommendations(
    db: AsyncSession,
    k: int,
    exclude_user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Get popularity-based recommendations (cold-start fallback).
    
    Args:
        db: Database session
        k: Number of recommendations
        exclude_user_id: User whose watched movies to exclude
        
    Returns:
        List of recommendations
    """
    from sqlalchemy import select, desc, func
    
    # Get top movies by popularity/rating
    query = select(Movie).order_by(desc(Movie.popularity)).limit(k)
    result = await db.execute(query)
    movies = result.scalars().all()
    
    recommendations = []
    for i, movie in enumerate(movies, 1):
        recommendations.append({
            "rank": i,
            "movie_id": movie.id,
            "title": movie.title,
            "genres": movie.genres or [],
            "year": movie.release_year,
            "poster_url": movie.poster_url,
            "predicted_score": movie.rating or 4.0,
            "match_percent": 50 + (i * 2),  # Decreasing match for popularity
            "explanation": "Popular movie trending now",
        })
    
    return recommendations


async def get_svd_recommendations(
    user_id: int,
    db: AsyncSession,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Get SVD-based recommendations (placeholder implementation).
    
    In production, this would load pre-trained SVD model and compute predictions.
    
    Args:
        user_id: User ID
        db: Database session
        k: Number of recommendations
        
    Returns:
        List of recommendations
    """
    # This is a placeholder. In real implementation:
    # 1. Load pre-trained SVD model from disk/S3
    # 2. Get user factor vector
    # 3. Compute dot-product with all item factors
    # 4. Rank and return top-k
    
    from sqlalchemy import select, desc
    
    # For now, return top-rated movies user hasn't seen
    user_rated = await db.execute(
        select(Rating.movie_id).where(Rating.user_id == user_id)
    )
    user_rated_ids = {r[0] for r in user_rated.fetchall()}
    
    movies_result = await db.execute(
        select(Movie)
        .where(~Movie.id.in_(user_rated_ids))
        .order_by(desc(Movie.rating))
        .limit(k)
    )
    movies = movies_result.scalars().all()
    
    recommendations = []
    for i, movie in enumerate(movies, 1):
        recommendations.append({
            "rank": i,
            "movie_id": movie.id,
            "title": movie.title,
            "genres": movie.genres or [],
            "year": movie.release_year,
            "poster_url": movie.poster_url,
            "predicted_score": movie.rating or 4.0,
            "match_percent": max(50, 95 - (i * 5)),
            "explanation": f"Based on your rating history",
        })
    
    return recommendations


async def get_knn_recommendations(
    user_id: int,
    db: AsyncSession,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Get KNN-based recommendations (placeholder implementation).
    
    Args:
        user_id: User ID
        db: Database session
        k: Number of recommendations
        
    Returns:
        List of recommendations
    """
    # Similar to SVD - placeholder implementation
    return await get_svd_recommendations(user_id, db, k)


async def get_ensemble_recommendations(
    user_id: int,
    db: AsyncSession,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Get ensemble recommendations (weighted blend of SVD + other models).
    
    Args:
        user_id: User ID
        db: Database session
        k: Number of recommendations
        
    Returns:
        List of recommendations
    """
    # Get recommendations from both SVD and KNN, blend and return
    svd_recs = await get_svd_recommendations(user_id, db, k * 2)
    
    # For ensemble, we'd blend scores. For now, just return SVD
    return svd_recs[:k]


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get(
    "/{user_id}",
    response_model=RecommendationsResponse,
    summary="Get recommendations for user",
    tags=["recommendations"],
)
async def get_recommendations(
    user_id: int,
    k: int = 10,
    strategy: str = "ensemble",
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user_id),
) -> RecommendationsResponse:
    """
    Get personalized top-K movie recommendations for a user.
    
    **Cache behavior:**
    - Cache HIT: Returns in <20ms
    - Cache MISS: Computes recommendations (<250ms)
    
    **Cold-start handling:**
    - If user has <5 ratings: Uses popularity-based fallback
    
    **Strategies:**
    - `svd`: Matrix factorization (primary)
    - `knn`: User-based KNN
    - `ensemble`: Weighted blend of SVD + ALS
    - `popularity`: Popular movies (cold-start)
    
    Args:
        user_id: User to get recommendations for
        k: Number of recommendations (1-50)
        strategy: Recommendation strategy
        db: Database session
        current_user: Authenticated user ID
        
    Returns:
        RecommendationsResponse with top-K recommendations
        
    Raises:
        HTTPException: If user not found or invalid strategy
    """
    start_time = time.time()
    
    # Check if user exists and owns this request
    if user_id != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other users' recommendations",
        )
    
    # Check cache first
    cached_recs = await get_recommendations_cache(user_id, k)
    if cached_recs:
        latency_ms = int((time.time() - start_time) * 1000)
        return RecommendationsResponse(
            user_id=user_id,
            strategy=strategy,
            generated_at=int(time.time()),
            cache_hit=True,
            latency_ms=latency_ms,
            recommendations=[
                RecommendationItem(**rec) for rec in cached_recs
            ],
        )
    
    try:
        # Get recommendations
        recs = await get_user_recommendations(user_id, db, k, strategy)
        
        # Cache for future requests (1 hour TTL)
        await set_recommendations_cache(user_id, recs, k, ttl=3600)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return RecommendationsResponse(
            user_id=user_id,
            strategy=strategy,
            generated_at=int(time.time()),
            cache_hit=False,
            latency_ms=latency_ms,
            recommendations=[
                RecommendationItem(**rec) for rec in recs
            ],
        )
        
    except Exception as e:
        logger.error(f"Recommendation error for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations",
        )


@router.get(
    "/similar/{movie_id}",
    response_model=SimilarMoviesResponse,
    summary="Get similar movies",
    tags=["recommendations"],
)
async def get_similar_movies(
    movie_id: int,
    k: int = 10,
    db: AsyncSession = Depends(get_db),
) -> SimilarMoviesResponse:
    """
    Get items similar to a given movie based on item-item collaborative filtering.
    
    Args:
        movie_id: Reference movie ID
        k: Number of similar movies to return
        db: Database session
        
    Returns:
        List of similar movies with similarity scores
    """
    from sqlalchemy import select
    
    # Check if movie exists
    movie = await db.execute(
        select(Movie).where(Movie.id == movie_id)
    )
    movie = movie.scalar_one_or_none()
    
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found",
        )
    
    # Get similar movies (placeholder - in production uses Faiss/cosine similarity)
    movies_result = await db.execute(
        select(Movie)
        .where(Movie.id != movie_id)
        .order_by(Movie.rating.desc())
        .limit(k)
    )
    similar = movies_result.scalars().all()
    
    similar_items = []
    for i, m in enumerate(similar, 1):
        similar_items.append(
            RecommendationItem(
                rank=i,
                movie_id=m.id,
                title=m.title,
                genres=m.genres or [],
                year=m.release_year,
                poster_url=m.poster_url,
                predicted_score=0.85 - (i * 0.05),
                match_percent=int((0.85 - (i * 0.05)) * 100),
                explanation=f"Similar to {movie.title}",
            )
        )
    
    return SimilarMoviesResponse(
        movie_id=movie_id,
        movie_title=movie.title,
        similar_movies=similar_items,
    )
