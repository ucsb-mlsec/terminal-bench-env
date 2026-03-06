#!/usr/bin/env python3

import redis
import json


class CacheService:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis connection for caching service."""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise Exception(f"Failed to connect to Redis: {e}")
    
    def _generate_cache_key(self, query_type, **params):
        """
        Generate a cache key for the query.
        BUG: This implementation doesn't properly handle parameter ordering,
        causing cache collisions for queries with same values but different parameter names.
        """
        # BUG: Using str(params.values()) which doesn't preserve parameter names
        # and has inconsistent ordering, causing different queries to generate same keys
        param_str = str(params.values())
        cache_key = f"{query_type}_{param_str}"
        return cache_key
    
    def get_cached_query(self, query_type, **params):
        """
        Retrieve cached query results from Redis.
        
        Args:
            query_type: Type/name of the query
            **params: Query parameters
            
        Returns:
            Cached result if exists, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(query_type, **params)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
        except redis.RedisError as e:
            print(f"Redis error during cache retrieval: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding cached data: {e}")
            return None
    
    def cache_query_result(self, query_type, result, **params):
        """
        Store query results in Redis cache.
        
        Args:
            query_type: Type/name of the query
            result: Query result to cache
            **params: Query parameters
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(query_type, **params)
            serialized_result = json.dumps(result)
            
            # Set cache with 1 hour expiration
            self.redis_client.setex(
                cache_key,
                3600,
                serialized_result
            )
            return True
        except redis.RedisError as e:
            print(f"Redis error during cache storage: {e}")
            return False
        except (TypeError, ValueError) as e:
            print(f"Error serializing result: {e}")
            return False
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            self.redis_client.flushdb()
            return True
        except redis.RedisError as e:
            print(f"Redis error during cache clear: {e}")
            return False


if __name__ == "__main__":
    # Basic usage example
    cache = CacheService()
    
    # Cache a query result
    cache.cache_query_result(
        "get_user",
        {"id": 123, "name": "John", "email": "john@example.com"},
        user_id=123
    )
    
    # Retrieve cached result
    result = cache.get_cached_query("get_user", user_id=123)
    print(f"Cached result: {result}")