"""
Events API endpoints for logging user interactions (clicks, watches, ratings).
"""

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db, UserEvent, Rating, Movie, User
from app.core.security import get_current_user_id
from app.db.redis_client import invalidate_user_cache

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class EventRequest(BaseModel):
    """User event request"""
    event_type: str  # click, view, watch, add_to_list, remove_from_list
    movie_id: int
    duration_seconds: int = None
    metadata: Dict[str, Any] = None


class RatingRequest(BaseModel):
    """User rating request"""
    movie_id: int
    rating: float  # 0.5 to 5.0


class EventResponse(BaseModel):
    """Event response"""
    id: int
    user_id: int
    movie_id: int
    event_type: str
    timestamp: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/",
    response_model=EventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log user event",
)
async def log_event(
    request: EventRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id),
) -> EventResponse:
    """
    Log a user interaction event (click, view, watch, etc.).
    
    **Event Types:**
    - `click`: User clicked on a recommendation
    - `view`: User viewed movie details page
    - `watch`: User watched movie (requires duration_seconds)
    - `add_to_list`: Added to watchlist/favorites
    - `remove_from_list`: Removed from watchlist/favorites
    
    Args:
        request: Event details
        db: Database session
        current_user_id: Authenticated user ID
        
    Returns:
        Created event record
        
    Raises:
        HTTPException: If movie not found
    """
    # Verify movie exists
    movie_result = await db.execute(
        select(Movie).where(Movie.id == request.movie_id)
    )
    
    if not movie_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found",
        )
    
    # Create event
    event = UserEvent(
        user_id=current_user_id,
        movie_id=request.movie_id,
        event_type=request.event_type,
        duration_seconds=request.duration_seconds,
        metadata=request.metadata or {},
    )
    
    db.add(event)
    await db.commit()
    await db.refresh(event)
    
    logger.info(
        f"Event logged: user={current_user_id}, "
        f"event={request.event_type}, movie={request.movie_id}"
    )
    
    return EventResponse.model_validate(event)


@router.post(
    "/rate",
    response_model=Dict[str, Any],
    summary="Rate a movie",
)
async def rate_movie(
    request: RatingRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Submit or update a movie rating.
    
    If user has already rated the movie, the rating is updated.
    
    Args:
        request: Rating details
        db: Database session
        current_user_id: Authenticated user ID
        
    Returns:
        Rating record
        
    Raises:
        HTTPException: If movie not found or invalid rating
    """
    # Validate rating
    if request.rating < 0.5 or request.rating > 5.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Rating must be between 0.5 and 5.0",
        )
    
    # Verify movie exists
    movie_result = await db.execute(
        select(Movie).where(Movie.id == request.movie_id)
    )
    
    if not movie_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found",
        )
    
    # Check if rating already exists
    rating_result = await db.execute(
        select(Rating).where(
            (Rating.user_id == current_user_id) &
            (Rating.movie_id == request.movie_id)
        )
    )
    existing_rating = rating_result.scalar_one_or_none()
    
    if existing_rating:
        # Update existing rating
        existing_rating.rating = request.rating
        await db.commit()
        await db.refresh(existing_rating)
        logger.info(
            f"Rating updated: user={current_user_id}, "
            f"movie={request.movie_id}, rating={request.rating}"
        )
    else:
        # Create new rating
        rating = Rating(
            user_id=current_user_id,
            movie_id=request.movie_id,
            rating=request.rating,
        )
        db.add(rating)
        await db.commit()
        await db.refresh(rating)
        existing_rating = rating
        logger.info(
            f"Rating created: user={current_user_id}, "
            f"movie={request.movie_id}, rating={request.rating}"
        )
    
    # Invalidate user's cached recommendations
    await invalidate_user_cache(current_user_id)
    
    return {
        "user_id": existing_rating.user_id,
        "movie_id": existing_rating.movie_id,
        "rating": existing_rating.rating,
        "timestamp": existing_rating.timestamp.isoformat(),
        "message": "Rating submitted successfully",
    }


@router.get(
    "/history",
    summary="Get user event history",
)
async def get_event_history(
    limit: int = 50,
    offset: int = 0,
    event_type: str = None,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Get user's event history (clicks, watches, ratings).
    
    Args:
        limit: Number of events to return
        offset: Number of events to skip
        event_type: Filter by event type (optional)
        db: Database session
        current_user_id: Authenticated user ID
        
    Returns:
        User's event history
    """
    query = select(UserEvent).where(UserEvent.user_id == current_user_id)
    
    if event_type:
        query = query.where(UserEvent.event_type == event_type)
    
    # Get total count
    count_result = await db.execute(
        select(UserEvent).where(UserEvent.user_id == current_user_id)
    )
    total = len(count_result.scalars().all())
    
    # Get paginated results
    result = await db.execute(
        query.order_by(UserEvent.timestamp.desc())
        .offset(offset)
        .limit(limit)
    )
    events = result.scalars().all()
    
    return {
        "user_id": current_user_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "events": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "movie_id": e.movie_id,
                "duration_seconds": e.duration_seconds,
                "timestamp": e.timestamp.isoformat(),
                "metadata": e.metadata,
            }
            for e in events
        ],
    }


@router.get(
    "/ratings",
    summary="Get user ratings",
)
async def get_user_ratings(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Get all ratings submitted by user.
    
    Args:
        limit: Number of ratings to return
        offset: Number of ratings to skip
        db: Database session
        current_user_id: Authenticated user ID
        
    Returns:
        User's ratings
    """
    # Get total count
    count_result = await db.execute(
        select(Rating).where(Rating.user_id == current_user_id)
    )
    total = len(count_result.scalars().all())
    
    # Get paginated results
    result = await db.execute(
        select(Rating)
        .where(Rating.user_id == current_user_id)
        .order_by(Rating.timestamp.desc())
        .offset(offset)
        .limit(limit)
    )
    ratings = result.scalars().all()
    
    return {
        "user_id": current_user_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "ratings": [
            {
                "movie_id": r.movie_id,
                "movie_title": r.movie.title if r.movie else None,
                "rating": r.rating,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in ratings
        ],
    }


@router.delete(
    "/ratings/{movie_id}",
    summary="Delete a rating",
)
async def delete_rating(
    movie_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Delete a rating for a movie.
    
    Args:
        movie_id: Movie ID
        db: Database session
        current_user_id: Authenticated user ID
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If rating not found
    """
    # Find and delete rating
    result = await db.execute(
        select(Rating).where(
            (Rating.user_id == current_user_id) &
            (Rating.movie_id == movie_id)
        )
    )
    rating = result.scalar_one_or_none()
    
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rating not found",
        )
    
    await db.delete(rating)
    await db.commit()
    
    # Invalidate cache
    await invalidate_user_cache(current_user_id)
    
    logger.info(f"Rating deleted: user={current_user_id}, movie={movie_id}")
    
    return {
        "message": "Rating deleted successfully",
        "movie_id": movie_id,
    }
