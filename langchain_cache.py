"""
LangChain Cache Manager for version 1.2.8
FIXED: No circular imports, all functions exported
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Redis for caching
try:
    import redis
    REDIS_AVAILABLE = True
    logger.info("✅ Redis module available")
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis package not installed. Using in-memory cache.")

# Cache manager class
class CacheManager:
    """Cache Manager for LangChain 1.2.8"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = None
        self.cache_enabled = False
        self.host = host
        self.port = port
        self.db = db
        self.cache_hits = 0
        self.cache_misses = 0
        self.langchain_cache_configured = False
        
        self.initialize_cache()
        # Don't setup LangChain cache here to avoid circular imports
    
    def initialize_cache(self) -> bool:
        """Initialize Redis connection"""
        try:
            if REDIS_AVAILABLE:
                self.redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3
                )
                
                self.redis_client.ping()
                logger.info(f"✅ Redis cache connected to {self.host}:{self.port}")
                self.cache_enabled = True
                return True
            else:
                logger.warning("⚠️ Redis package not installed. Using in-memory cache.")
                self.cache_enabled = True
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Redis cache not available: {e}")
            self.cache_enabled = True
            logger.info("✅ In-memory cache enabled (fallback)")
            return True
    
    def setup_langchain_cache(self):
        """Setup LangChain cache - call this separately to avoid circular imports"""
        try:
            # For LangChain 1.2.8
            from langchain_cache import set_llm_cache
            
            if self.redis_client and self.is_connected():
                try:
                    from langchain_community.cache import RedisCache
                    redis_cache = RedisCache(redis_=self.redis_client)
                    set_llm_cache(redis_cache)
                    logger.info("✅ LangChain Redis cache configured")
                    self.langchain_cache_configured = True
                except ImportError:
                    # Fallback to in-memory
                    from langchain_core.caches import InMemoryCache
                    in_memory_cache = InMemoryCache()
                    set_llm_cache(in_memory_cache)
                    logger.info("✅ LangChain in-memory cache configured (Redis import failed)")
                    self.langchain_cache_configured = True
            else:
                from langchain_core.caches import InMemoryCache
                in_memory_cache = InMemoryCache()
                set_llm_cache(in_memory_cache)
                logger.info("✅ LangChain in-memory cache configured")
                self.langchain_cache_configured = True
                
        except ImportError as e:
            logger.error(f"❌ Failed to setup LangChain cache: {e}")
            self.langchain_cache_configured = False
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if self.redis_client:
            try:
                return self.redis_client.ping()
            except:
                return False
        return self.cache_enabled
    
    def generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a consistent cache key from parameters"""
        sorted_kwargs = sorted(kwargs.items())
        params_str = json.dumps(sorted_kwargs, sort_keys=True)
        
        hash_obj = hashlib.md5(params_str.encode())
        hash_digest = hash_obj.hexdigest()
        
        return f"{prefix}:{hash_digest}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client and self.is_connected():
                value = self.redis_client.get(key)
                if value:
                    try:
                        result = json.loads(value)
                        self.cache_hits += 1
                        return result
                    except:
                        self.cache_hits += 1
                        return value
                self.cache_misses += 1
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            if self.redis_client and self.is_connected():
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value)
                    self.redis_client.setex(key, ttl, value_str)
                else:
                    self.redis_client.setex(key, ttl, str(value))
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client and self.is_connected():
                return self.redis_client.delete(key) > 0
            return False
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        try:
            if self.redis_client and self.is_connected():
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
        except Exception as e:
            logger.error(f"Error clearing pattern: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "cache_enabled": self.cache_enabled,
            "redis_connected": self.is_connected(),
            "langchain_cache_configured": self.langchain_cache_configured,
            "cache_type": "redis" if self.redis_client and self.is_connected() else "in_memory",
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": round(self.cache_hits / max(self.cache_hits + self.cache_misses, 1), 2),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.redis_client and self.is_connected():
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_used_memory": info.get('used_memory_human', 'N/A'),
                    "redis_keys": self.redis_client.dbsize(),
                    "redis_connections": info.get('connected_clients', 0)
                })
            except:
                stats["redis_info"] = "unavailable"
        
        return stats

# Create global cache manager instance
cache_manager = CacheManager()

# Helper functions that other modules can import
def get_cache_stats() -> Dict[str, Any]:
    return cache_manager.get_stats()

def cache_clear_pattern(pattern: str) -> int:
    return cache_manager.clear_pattern(pattern)

def is_redis_connected() -> bool:
    return cache_manager.is_connected()

def get_cache_manager() -> CacheManager:
    return cache_manager

def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key - exported for other modules"""
    return cache_manager.generate_cache_key(prefix, **kwargs)

def cache_get(key: str) -> Optional[Any]:
    return cache_manager.get(key)

def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    return cache_manager.set(key, value, ttl)

def cache_delete(key: str) -> bool:
    return cache_manager.delete(key)
def get_redis_client():
    """Get Redis client for other modules (for backward compatibility)"""
    return cache_manager.redis_client