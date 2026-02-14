"""Redis cache connection and management."""
import os
import json
import redis
import logging

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    # Parse Redis URL
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    cache = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=0,
        decode_responses=True
    )
    # Test connection
    cache.ping()
    logger.info("Redis cache connection initialized")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
    cache = None

def get_cached(key: str):
    """Get value from cache."""
    if cache is None:
        return None
    try:
        value = cache.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    return None

def set_cached(key: str, value, ttl: int = 3600):
    """Set value in cache with TTL."""
    if cache is None:
        return False
    try:
        cache.setex(key, ttl, json.dumps(value))
        return True
    except Exception as e:
        logger.warning(f"Cache set error: {e}")
    return False
