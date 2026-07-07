"""
Redis client initialization and utilities.

Provides connection pooling, caching utilities, and async Redis operations.
"""

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global Redis client
redis_client: Optional[aioredis.Redis] = None


async def init_redis() -> aioredis.Redis:
    """
    Initialize Redis connection pool.
    
    Returns:
        aioredis.Redis: Redis client instance
    """
    global redis_client
    
    try:
        redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf8",
            decode_responses=True,
            max_connections=settings.REDIS_POOL_SIZE,
            retry_on_timeout=True,
        )
        
        # Test connection
        await redis_client.ping()
        logger.info("✓ Redis connection established successfully")
        return redis_client
        
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        raise


async def get_redis() -> aioredis.Redis:
    """
    Get Redis client instance.
    
    Returns:
        aioredis.Redis: Redis client
        
    Raises:
        RuntimeError: If Redis not initialized
    """
    if redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return redis_client


# ============================================================================
# CACHING UTILITIES
# ============================================================================


async def cache_get(key: str) -> Optional[Any]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    try:
        client = await get_redis()
        value = await client.get(key)
        
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
        
    except RedisError as e:
        logger.error(f"Cache get error: {e}")
        return None


async def cache_set(
    key: str,
    value: Any,
    ttl: int = None,
) -> bool:
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache (will be JSON serialized if dict/list)
        ttl: Time-to-live in seconds (uses default if None)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = await get_redis()
        ttl = ttl or settings.REDIS_CACHE_TTL
        
        # Serialize if needed
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        await client.setex(key, ttl, value)
        return True
        
    except RedisError as e:
        logger.error(f"Cache set error: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """
    Delete key from cache.
    
    Args:
        key: Cache key
        
    Returns:
        True if successful
    """
    try:
        client = await get_redis()
        await client.delete(key)
        return True
    except RedisError as e:
        logger.error(f"Cache delete error: {e}")
        return False


async def cache_exists(key: str) -> bool:
    """
    Check if key exists in cache.
    
    Args:
        key: Cache key
        
    Returns:
        True if exists
    """
    try:
        client = await get_redis()
        return await client.exists(key)
    except RedisError as e:
        logger.error(f"Cache exists error: {e}")
        return False


# ============================================================================
# RECOMMENDATION CACHE
# ============================================================================


async def get_recommendations_cache(user_id: int, k: int = 10) -> Optional[list]:
    """
    Get cached recommendations for user.
    
    Args:
        user_id: User ID
        k: Number of recommendations
        
    Returns:
        List of recommendations or None
    """
    key = f"rec:{user_id}:{k}"
    return await cache_get(key)


async def set_recommendations_cache(
    user_id: int,
    recommendations: list,
    k: int = 10,
    ttl: int = None,
) -> bool:
    """
    Cache recommendations for user.
    
    Args:
        user_id: User ID
        recommendations: List of recommendation dicts
        k: Number of recommendations
        ttl: Cache TTL in seconds
        
    Returns:
        True if successful
    """
    key = f"rec:{user_id}:{k}"
    return await cache_set(key, recommendations, ttl)


async def invalidate_user_cache(user_id: int) -> bool:
    """
    Invalidate all cached recommendations for user.
    
    Args:
        user_id: User ID
        
    Returns:
        True if successful
    """
    try:
        client = await get_redis()
        pattern = f"rec:{user_id}:*"
        keys = await client.keys(pattern)
        
        if keys:
            await client.delete(*keys)
        
        return True
    except RedisError as e:
        logger.error(f"Cache invalidation error: {e}")
        return False


# ============================================================================
# SESSION & RATE LIMITING
# ============================================================================


async def increment_rate_limit(user_id: int, window: int = 60) -> int:
    """
    Increment rate limit counter for user.
    
    Args:
        user_id: User ID
        window: Time window in seconds
        
    Returns:
        Current count
    """
    try:
        client = await get_redis()
        key = f"rate:{user_id}"
        count = await client.incr(key)
        
        if count == 1:
            await client.expire(key, window)
        
        return count
    except RedisError as e:
        logger.error(f"Rate limit error: {e}")
        return 0


async def get_rate_limit(user_id: int) -> int:
    """
    Get current rate limit count.
    
    Args:
        user_id: User ID
        
    Returns:
        Current count
    """
    try:
        client = await get_redis()
        key = f"rate:{user_id}"
        count = await client.get(key)
        return int(count) if count else 0
    except RedisError as e:
        logger.error(f"Rate limit get error: {e}")
        return 0


# ============================================================================
# HEALTH CHECKS
# ============================================================================


async def check_redis_health() -> dict:
    """
    Check Redis health and connectivity.
    
    Returns:
        Health status dict
    """
    try:
        if redis_client is None:
            return {
                "status": "disconnected",
                "error": "Redis client not initialized",
            }
        
        info = await redis_client.info()
        return {
            "status": "healthy",
            "uptime_seconds": info.get("uptime_in_seconds"),
            "connected_clients": info.get("connected_clients"),
            "used_memory": info.get("used_memory_human"),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
