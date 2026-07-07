"""
CineMatch AI - FastAPI Application Entry Point

Production-grade FastAPI application with:
- Async database setup
- Redis caching
- JWT authentication
- CORS configuration
- Health checks
- Error handling
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.security import setup_security
from app.db.database import init_db, get_db
from app.db.redis_client import init_redis, redis_client
from app.api import recommendations, movies, auth, events, health

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for startup and shutdown events.
    
    Startup:
    - Initialize database connection pool
    - Initialize Redis connection
    - Setup security configurations
    
    Shutdown:
    - Close database connections
    - Close Redis connections
    """
    # Startup
    logger.info("🚀 Starting CineMatch AI API...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("✓ Database initialized successfully")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise
    
    try:
        # Initialize Redis
        await init_redis()
        logger.info("✓ Redis cache initialized successfully")
    except Exception as e:
        logger.error(f"✗ Redis initialization failed: {e}")
        raise
    
    logger.info("✓ CineMatch AI API is ready to serve requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down CineMatch AI API...")
    
    try:
        if redis_client:
            await redis_client.close()
        logger.info("✓ Redis connection closed")
    except Exception as e:
        logger.error(f"✗ Error closing Redis: {e}")
    
    logger.info("✓ CineMatch AI API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description="Collaborative Filtering Movie Recommendation Engine",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
    max_age=3600,
)

# Trusted Host Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.cinematch.ai"],
)

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

# Health check routes
app.include_router(
    health.router,
    prefix="/health",
    tags=["health"],
)

# Authentication routes
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["authentication"],
)

# Recommendation routes
app.include_router(
    recommendations.router,
    prefix="/api/v1/recommendations",
    tags=["recommendations"],
)

# Movie routes
app.include_router(
    movies.router,
    prefix="/api/v1/movies",
    tags=["movies"],
)

# Event logging routes
app.include_router(
    events.router,
    prefix="/api/v1/events",
    tags=["events"],
)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["root"])
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "CineMatch AI",
        "version": settings.API_VERSION,
        "status": "healthy",
        "docs": "/docs",
        "environment": settings.ENVIRONMENT,
    }


# ============================================================================
# DEBUG & DEVELOPMENT ENDPOINTS
# ============================================================================

if settings.ENVIRONMENT == "development":
    
    @app.get("/api/v1/debug/config", tags=["debug"])
    async def get_config():
        """Get current configuration (development only)"""
        return {
            "environment": settings.ENVIRONMENT,
            "api_title": settings.API_TITLE,
            "api_version": settings.API_VERSION,
            "database": "configured" if settings.DATABASE_URL else "not configured",
            "redis": "configured" if settings.REDIS_URL else "not configured",
            "cors_origins": settings.ALLOWED_ORIGINS,
        }
    
    @app.get("/api/v1/debug/redis", tags=["debug"])
    async def check_redis():
        """Check Redis connectivity (development only)"""
        try:
            pong = await redis_client.ping()
            return {
                "status": "connected",
                "response": pong,
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "error": str(e),
            }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT == "development",
        workers=1 if settings.ENVIRONMENT == "development" else settings.API_WORKERS,
    )
