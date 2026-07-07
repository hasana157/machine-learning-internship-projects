"""
Movies API endpoints for catalog, search, and details.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, or_

from app.db.database import get_db, Movie, Rating, User

logger = logging.getLogger(__name__)

router = APIRouter()


class MovieDetail:
    """Movie detail response"""
    def __init__(self, movie: Movie, user_rating: Optional[float] = None):
        self.id = movie.id
        self.title = movie.title
        self.release_year = movie.release_year
        self.overview = movie.overview
        self.poster_url = movie.poster_url
        self.rating = movie.rating
        self.runtime = movie.runtime
        self.genres = movie.genres or []
        self.popularity = movie.popularity
        self.user_rating = user_rating


@router.get(
    "/",
    summary="List movies",
    tags=["movies"],
)
async def list_movies(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("popularity", regex="^(popularity|rating|title|year)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    List movies from catalog with pagination and sorting.
    
    Args:
        skip: Number of movies to skip
        limit: Number of movies to return
        sort_by: Sort field (popularity, rating, title, year)
        db: Database session
        
    Returns:
        List of movies
    """
    # Build query
    query = select(Movie)
    
    # Add sorting
    if sort_by == "popularity":
        query = query.order_by(desc(Movie.popularity))
    elif sort_by == "rating":
        query = query.order_by(desc(Movie.rating))
    elif sort_by == "title":
        query = query.order_by(Movie.title)
    elif sort_by == "year":
        query = query.order_by(desc(Movie.release_year))
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    movies = result.scalars().all()
    
    return {
        "items": [
            {
                "id": m.id,
                "title": m.title,
                "year": m.release_year,
                "poster_url": m.poster_url,
                "rating": m.rating,
                "genres": m.genres or [],
                "popularity": m.popularity,
            }
            for m in movies
        ],
        "skip": skip,
        "limit": limit,
        "total": len(movies),
    }


@router.get(
    "/search",
    summary="Search movies",
    tags=["movies"],
)
async def search_movies(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Search movies by title and overview.
    
    Args:
        q: Search query
        limit: Maximum results
        db: Database session
        
    Returns:
        Search results
    """
    # Build search query
    search_term = f"%{q}%"
    
    result = await db.execute(
        select(Movie)
        .where(
            or_(
                Movie.title.ilike(search_term),
                Movie.overview.ilike(search_term),
            )
        )
        .limit(limit)
    )
    movies = result.scalars().all()
    
    return {
        "query": q,
        "results": [
            {
                "id": m.id,
                "title": m.title,
                "year": m.release_year,
                "poster_url": m.poster_url,
                "rating": m.rating,
                "genres": m.genres or [],
            }
            for m in movies
        ],
        "count": len(movies),
    }


@router.get(
    "/{movie_id}",
    summary="Get movie details",
    tags=["movies"],
)
async def get_movie_detail(
    movie_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a movie.
    
    Args:
        movie_id: Movie ID
        db: Database session
        
    Returns:
        Movie details with ratings count
        
    Raises:
        HTTPException: If movie not found
    """
    # Get movie
    result = await db.execute(
        select(Movie).where(Movie.id == movie_id)
    )
    movie = result.scalar_one_or_none()
    
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found",
        )
    
    # Get ratings count and average
    ratings_result = await db.execute(
        select(
            func.count(Rating.id).label("count"),
            func.avg(Rating.rating).label("avg_rating"),
        ).where(Rating.movie_id == movie_id)
    )
    ratings = ratings_result.one()
    
    return {
        "id": movie.id,
        "title": movie.title,
        "year": movie.release_year,
        "overview": movie.overview,
        "poster_url": movie.poster_url,
        "runtime": movie.runtime,
        "genres": movie.genres or [],
        "rating": movie.rating,
        "popularity": movie.popularity,
        "ratings_count": ratings[0] or 0,
        "average_rating": float(ratings[1]) if ratings[1] else None,
    }


@router.get(
    "/{movie_id}/recommendations",
    summary="Get recommendations based on movie",
    tags=["movies"],
)
async def get_movie_recommendations(
    movie_id: int,
    k: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Get similar movies (item-based recommendations).
    
    Args:
        movie_id: Reference movie ID
        k: Number of recommendations
        db: Database session
        
    Returns:
        List of similar movies
    """
    # Check if movie exists
    result = await db.execute(
        select(Movie).where(Movie.id == movie_id)
    )
    movie = result.scalar_one_or_none()
    
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found",
        )
    
    # Get similar movies (by genre for now)
    if movie.genres:
        similar_result = await db.execute(
            select(Movie)
            .where(Movie.id != movie_id)
            .order_by(desc(Movie.rating))
            .limit(k)
        )
        similar_movies = similar_result.scalars().all()
    else:
        similar_movies = []
    
    return {
        "reference_movie_id": movie_id,
        "reference_movie_title": movie.title,
        "similar_movies": [
            {
                "id": m.id,
                "title": m.title,
                "year": m.release_year,
                "poster_url": m.poster_url,
                "rating": m.rating,
                "genres": m.genres or [],
                "similarity_score": 0.85,  # Placeholder
            }
            for m in similar_movies
        ],
    }


@router.get(
    "/stats/trending",
    summary="Get trending movies",
    tags=["movies"],
)
async def get_trending_movies(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Get currently trending movies based on ratings and popularity.
    
    Args:
        limit: Number of movies to return
        db: Database session
        
    Returns:
        Trending movies
    """
    result = await db.execute(
        select(Movie)
        .order_by(desc(Movie.popularity))
        .limit(limit)
    )
    movies = result.scalars().all()
    
    return {
        "trending": [
            {
                "id": m.id,
                "title": m.title,
                "year": m.release_year,
                "poster_url": m.poster_url,
                "rating": m.rating,
                "popularity": m.popularity,
            }
            for m in movies
        ]
    }
