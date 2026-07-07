"""
Pydantic schemas for recommendation endpoints.

Defines request/response models for type safety and API documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class MovieResponse(BaseModel):
    """Movie response schema"""
    
    id: int
    title: str
    release_year: Optional[int] = None
    tmdb_id: Optional[int] = None
    poster_url: Optional[str] = None
    overview: Optional[str] = None
    rating: Optional[float] = None
    genres: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class RecommendationItem(BaseModel):
    """Individual recommendation item"""
    
    rank: int
    movie_id: int
    title: str
    genres: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    poster_url: Optional[str] = None
    predicted_score: float = Field(
        ...,
        description="Predicted rating (0-5)"
    )
    match_percent: int = Field(
        ...,
        description="Match percentage (0-100)"
    )
    explanation: Optional[str] = Field(
        None,
        description="Why this recommendation (e.g., 'Because you liked X')"
    )


class RecommendationsResponse(BaseModel):
    """Recommendations endpoint response"""
    
    user_id: int
    strategy: str = Field(
        ...,
        description="Recommendation strategy used: svd, knn, ensemble, popularity"
    )
    generated_at: datetime
    cache_hit: bool = Field(
        ...,
        description="Whether result was served from cache"
    )
    latency_ms: int = Field(
        ...,
        description="Response latency in milliseconds"
    )
    recommendations: List[RecommendationItem]
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "strategy": "ensemble",
                "generated_at": "2024-01-01T00:00:00",
                "cache_hit": False,
                "latency_ms": 150,
                "recommendations": [
                    {
                        "rank": 1,
                        "movie_id": 318,
                        "title": "The Shawshank Redemption",
                        "genres": ["Drama"],
                        "year": 1994,
                        "predicted_score": 4.87,
                        "match_percent": 97,
                        "explanation": "Because you rated The Green Mile 5.0"
                    }
                ]
            }
        }


class SimilarMoviesResponse(BaseModel):
    """Similar movies response"""
    
    movie_id: int
    movie_title: str
    similar_movies: List[RecommendationItem]
    strategy: str = "item_similarity"


class RecommendationRequest(BaseModel):
    """Recommendation request parameters"""
    
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to return"
    )
    strategy: str = Field(
        default="ensemble",
        description="Recommendation strategy: svd, knn, ensemble, popularity"
    )
    exclude_watched: bool = Field(
        default=True,
        description="Exclude already-watched movies"
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0-1)"
    )


class RatingRequest(BaseModel):
    """Rating submission request"""
    
    movie_id: int
    rating: float = Field(
        ...,
        ge=0.5,
        le=5.0,
        description="Rating value (0.5-5.0)"
    )


class RatingResponse(BaseModel):
    """Rating submission response"""
    
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime
    
    class Config:
        from_attributes = True


class RatingUpdate(BaseModel):
    """Update user rating"""
    
    rating: float = Field(
        ...,
        ge=0.5,
        le=5.0,
    )
