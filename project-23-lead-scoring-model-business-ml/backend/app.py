"""
LeadForge AI - Main Litestar API Application
Production-ready with proper error handling, logging, and async support
"""

import logging
import json
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime

from litestar import Litestar, post, get, MediaType
from litestar.di import Provide
from litestar.status_codes import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from litestar.exceptions import HTTPException, ValidationException
from litestar.serialization import encode_json
from pydantic import BaseModel, Field, validator

import redis
import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.ml_service import MLService
from services.feature_engineering import FeatureEngineer
from services.database_service import DatabaseService
from config import settings

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class LeadFeatures(BaseModel):
    """Input features for a single lead"""
    lead_id: str = Field(..., description="Unique lead identifier")
    company_size: Optional[int] = Field(None, ge=1, le=100000)
    industry: Optional[str] = Field(None, max_length=100)
    email_opens: Optional[int] = Field(None, ge=0, le=1000)
    email_clicks: Optional[int] = Field(None, ge=0, le=1000)
    days_since_contact: Optional[int] = Field(None, ge=0, le=365)
    deal_value: Optional[float] = Field(None, ge=0, le=10000000)
    product_interest: Optional[List[str]] = Field(None)
    engagement_score: Optional[float] = Field(None, ge=0, le=100)
    last_activity_type: Optional[str] = Field(None)
    
    @validator('lead_id')
    def validate_lead_id(cls, v):
        if not v or len(v) == 0:
            raise ValueError("lead_id cannot be empty")
        return v


class SHAPFeature(BaseModel):
    """SHAP feature importance"""
    name: str
    value: float
    impact: str


class ScoreResponse(BaseModel):
    """API response for lead scoring"""
    lead_id: str
    score: int = Field(..., ge=0, le=100)
    tier: str = Field(..., regex="^(hot|warm|cold)$")
    conversion_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_features: List[SHAPFeature]
    explainability: dict
    timestamp: str
    model_version: str


class BatchScoreRequest(BaseModel):
    """Request for batch lead scoring"""
    leads: List[LeadFeatures] = Field(..., min_items=1, max_items=10000)
    async_processing: bool = False


class BatchScoreResponse(BaseModel):
    """Response for batch scoring"""
    job_id: Optional[str] = None
    results: Optional[List[ScoreResponse]] = None
    status: str
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database: str
    redis: str
    model: str
    version: str
    timestamp: str


class ModelMetadataResponse(BaseModel):
    """Model metadata response"""
    model_id: str
    framework: str
    version: str
    metrics: dict
    training_date: str
    feature_count: int
    calibrated: bool


# ============================================================================
# Service Dependencies
# ============================================================================

def get_ml_service() -> MLService:
    """Dependency: ML Service"""
    return MLService(
        model_path=settings.MODEL_PATH,
        scaler_path=settings.SCALER_PATH,
        calibrator_path=settings.CALIBRATOR_PATH,
        feature_names_path=settings.FEATURE_NAMES_PATH
    )


def get_feature_engineer() -> FeatureEngineer:
    """Dependency: Feature Engineer"""
    return FeatureEngineer()


async def get_redis_client() -> redis.asyncio.Redis:
    """Dependency: Redis client"""
    try:
        client = await redis.asyncio.from_url(
            settings.REDIS_URL,
            encoding="utf8",
            decode_responses=True,
            socket_connect_timeout=5
        )
        await client.ping()
        return client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Redis service unavailable"
        )


async def get_db_service() -> DatabaseService:
    """Dependency: Database service"""
    try:
        service = DatabaseService(database_url=settings.DATABASE_URL)
        await service.connect()
        return service
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database service unavailable"
        )


# ============================================================================
# Route Handlers
# ============================================================================

@get(
    "/api/v1/health",
    status_code=HTTP_200_OK,
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check(
    redis_client: redis.asyncio.Redis,
    db_service: DatabaseService,
    ml_service: MLService
) -> HealthResponse:
    """
    Health check endpoint
    Verifies all critical services are operational
    """
    try:
        # Check Redis
        redis_status = "healthy"
        try:
            await redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            redis_status = "unhealthy"
        
        # Check Database
        db_status = "healthy"
        try:
            await db_service.health_check()
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            db_status = "unhealthy"
        
        # Check Model
        model_status = "healthy" if ml_service.is_loaded else "not_loaded"
        
        # Determine overall status
        overall_status = "healthy" if all([
            redis_status == "healthy",
            db_status == "healthy",
            model_status == "healthy"
        ]) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            database=db_status,
            redis=redis_status,
            model=model_status,
            version=settings.API_VERSION,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            database="unknown",
            redis="unknown",
            model="unknown",
            version=settings.API_VERSION,
            timestamp=datetime.utcnow().isoformat()
        )


@post(
    "/api/v1/leads/score",
    status_code=HTTP_200_OK,
    response_model=ScoreResponse,
    tags=["Lead Scoring"]
)
async def score_single_lead(
    data: LeadFeatures,
    ml_service: MLService,
    feature_engineer: FeatureEngineer,
    db_service: DatabaseService,
    redis_client: redis.asyncio.Redis
) -> ScoreResponse:
    """
    Score a single lead with explainability
    Returns conversion probability, tier, and SHAP feature importance
    """
    try:
        logger.info(f"Scoring lead: {data.lead_id}")
        
        # Check cache first
        cache_key = f"lead_score:{data.lead_id}"
        cached = await redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for {data.lead_id}")
            return ScoreResponse(**json.loads(cached))
        
        # Transform features
        features_dict = data.model_dump(exclude_none=True)
        X = feature_engineer.transform_single(features_dict)
        
        # Get prediction and explainability
        prediction = ml_service.predict(X)
        shap_values = ml_service.get_shap_values(X)
        probability = ml_service.predict_proba(X)
        confidence = ml_service.get_confidence(X)
        
        # Determine tier
        if probability >= 0.7:
            tier = "hot"
        elif probability >= 0.4:
            tier = "warm"
        else:
            tier = "cold"
        
        # Format SHAP features
        feature_names = ml_service.feature_names
        top_features = []
        if shap_values is not None:
            # Get top 3 most impactful features
            top_indices = sorted(
                range(len(shap_values)),
                key=lambda i: abs(shap_values[i]),
                reverse=True
            )[:3]
            
            for idx in top_indices:
                top_features.append(SHAPFeature(
                    name=feature_names[idx],
                    value=float(shap_values[idx]),
                    impact="positive" if shap_values[idx] > 0 else "negative"
                ))
        
        # Build response
        response = ScoreResponse(
            lead_id=data.lead_id,
            score=int(probability * 100),
            tier=tier,
            conversion_probability=float(probability),
            confidence=float(confidence),
            top_features=top_features,
            explainability={
                "positive": [f.name for f in top_features if f.impact == "positive"],
                "negative": [f.name for f in top_features if f.impact == "negative"]
            },
            timestamp=datetime.utcnow().isoformat(),
            model_version=ml_service.model_version
        )
        
        # Save to database
        await db_service.save_score(
            lead_id=data.lead_id,
            score=response.score,
            probability=response.conversion_probability,
            tier=response.tier,
            features=features_dict
        )
        
        # Cache for 1 hour
        await redis_client.setex(
            cache_key,
            3600,
            response.model_dump_json()
        )
        
        logger.info(f"Successfully scored {data.lead_id}: {response.score}")
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error scoring lead: {e}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during lead scoring. Please try again."
        )


@post(
    "/api/v1/leads/batch_score",
    status_code=HTTP_200_OK,
    response_model=BatchScoreResponse,
    tags=["Lead Scoring"]
)
async def batch_score_leads(
    data: BatchScoreRequest,
    ml_service: MLService,
    feature_engineer: FeatureEngineer,
    db_service: DatabaseService,
    redis_client: redis.asyncio.Redis
) -> BatchScoreResponse:
    """
    Score multiple leads in batch
    Returns job_id if async=true, otherwise returns results immediately
    """
    try:
        logger.info(f"Batch scoring {len(data.leads)} leads")
        
        if data.async_processing:
            # Queue job for async processing
            job_id = await db_service.queue_batch_job(
                leads=[lead.model_dump() for lead in data.leads]
            )
            logger.info(f"Batch job queued: {job_id}")
            
            return BatchScoreResponse(
                job_id=job_id,
                status="queued",
                total_processed=0,
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            # Process synchronously
            results = []
            for lead in data.leads:
                try:
                    # Transform features
                    features_dict = lead.model_dump(exclude_none=True)
                    X = feature_engineer.transform_single(features_dict)
                    
                    # Get predictions
                    probability = ml_service.predict_proba(X)
                    confidence = ml_service.get_confidence(X)
                    
                    # Determine tier
                    if probability >= 0.7:
                        tier = "hot"
                    elif probability >= 0.4:
                        tier = "warm"
                    else:
                        tier = "cold"
                    
                    # Get SHAP values
                    shap_values = ml_service.get_shap_values(X)
                    top_features = []
                    if shap_values is not None:
                        top_indices = sorted(
                            range(len(shap_values)),
                            key=lambda i: abs(shap_values[i]),
                            reverse=True
                        )[:3]
                        
                        for idx in top_indices:
                            top_features.append(SHAPFeature(
                                name=ml_service.feature_names[idx],
                                value=float(shap_values[idx]),
                                impact="positive" if shap_values[idx] > 0 else "negative"
                            ))
                    
                    score_result = ScoreResponse(
                        lead_id=lead.lead_id,
                        score=int(probability * 100),
                        tier=tier,
                        conversion_probability=float(probability),
                        confidence=float(confidence),
                        top_features=top_features,
                        explainability={
                            "positive": [f.name for f in top_features if f.impact == "positive"],
                            "negative": [f.name for f in top_features if f.impact == "negative"]
                        },
                        timestamp=datetime.utcnow().isoformat(),
                        model_version=ml_service.model_version
                    )
                    results.append(score_result)
                    
                    # Save to database
                    await db_service.save_score(
                        lead_id=lead.lead_id,
                        score=score_result.score,
                        probability=score_result.conversion_probability,
                        tier=score_result.tier,
                        features=features_dict
                    )
                
                except Exception as e:
                    logger.error(f"Error scoring lead {lead.lead_id}: {e}")
                    continue
            
            logger.info(f"Successfully scored {len(results)} leads")
            
            return BatchScoreResponse(
                results=results,
                status="completed",
                total_processed=len(results),
                timestamp=datetime.utcnow().isoformat()
            )
    
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during batch scoring"
        )


@get(
    "/api/v1/models/current",
    status_code=HTTP_200_OK,
    response_model=ModelMetadataResponse,
    tags=["Models"]
)
async def get_model_metadata(
    ml_service: MLService
) -> ModelMetadataResponse:
    """
    Get current model metadata and performance metrics
    """
    try:
        metadata = ml_service.get_metadata()
        return ModelMetadataResponse(**metadata)
    except Exception as e:
        logger.error(f"Error retrieving model metadata: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve model metadata"
        )


# ============================================================================
# Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: Litestar) -> None:
    """Application lifecycle management"""
    logger.info("Starting LeadForge AI API")
    yield
    logger.info("Shutting down LeadForge AI API")


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> Litestar:
    """Create and configure Litestar application"""
    
    app = Litestar(
        route_handlers=[
            health_check,
            score_single_lead,
            batch_score_leads,
            get_model_metadata
        ],
        dependencies={
            "ml_service": Provide(get_ml_service, sync_to_thread=False),
            "feature_engineer": Provide(get_feature_engineer, sync_to_thread=False),
            "redis_client": Provide(get_redis_client),
            "db_service": Provide(get_db_service)
        },
        lifespan=[lifespan],
        debug=settings.DEBUG,
        cors_config={
            "allow_origins": settings.ALLOWED_ORIGINS,
            "allow_methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    )
    
    return app


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
