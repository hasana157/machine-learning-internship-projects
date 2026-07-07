"""
Health check endpoints for monitoring and diagnostics.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.database import get_db
from app.db.redis_client import check_redis_health

router = APIRouter()


@router.get("/", tags=["health"])
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}


@router.get("/live", tags=["health"])
async def liveness_probe():
    """Kubernetes liveness probe"""
    return {"status": "alive"}


@router.get("/ready", tags=["health"])
async def readiness_probe(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes readiness probe.
    
    Checks if application is ready to serve requests.
    """
    try:
        # Check database
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check Redis
    redis_status = await check_redis_health()
    
    is_ready = db_status == "healthy" and redis_status.get("status") == "healthy"
    
    return {
        "status": "ready" if is_ready else "not_ready",
        "database": db_status,
        "redis": redis_status.get("status"),
    }


@router.get("/detailed", tags=["health"])
async def detailed_health(db: AsyncSession = Depends(get_db)):
    """Get detailed health information"""
    
    # Database info
    try:
        result = await db.execute(text("SELECT version()"))
        db_version = result.scalar()
        db_status = "healthy"
        db_error = None
    except Exception as e:
        db_version = None
        db_status = "unhealthy"
        db_error = str(e)
    
    # Redis info
    redis_info = await check_redis_health()
    
    return {
        "status": "healthy",
        "database": {
            "status": db_status,
            "version": db_version,
            "error": db_error,
        },
        "redis": redis_info,
    }
