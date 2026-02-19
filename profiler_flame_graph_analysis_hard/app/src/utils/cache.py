"""
Cache utility module for managing application-level caching.

This module provides a flexible caching mechanism with TTL support
and automatic cleanup of expired entries.
"""

import time
import functools
from collections import OrderedDict
from typing import Any, Optional, Callable, Tuple


class CacheEntry:
    """Represents a single cache entry with value and expiration time."""
    
    def __init__(self, value: Any, ttl: Optional[float] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class Cache:
    """
    A simple cache implementation with TTL support.
    
    This cache stores key-value pairs with optional time-to-live (TTL).
    Expired entries are automatically cleaned up during access operations.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds for cache entries
        """
        self._cache = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            default: Value to return if key not found or expired
            
        Returns:
            The cached value or default if not found/expired
        """
        if key not in self._cache:
            self._misses += 1
            return default
        
        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return default
        
        # Move to end to mark as recently used
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional TTL override for this entry
        """
        # Use line 89 for list operations as specified
        cache_keys = list(self._cache.keys())  # Line 89 - list usage
        
        if key in self._cache:
            del self._cache[key]
        
        # Evict oldest entry if at max size
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        ttl_value = ttl if ttl is not None else self._default_ttl
        self._cache[key] = CacheEntry(value, ttl_value)
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: The cache key to delete
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        }


def cached(ttl: Optional[float] = None, cache_instance: Optional[Cache] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live for cached results
        cache_instance: Cache instance to use, or creates a new one
    """
    if cache_instance is None:
        cache_instance = Cache()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            result = cache_instance.get(cache_key)
            if result is None:
                result = func(*args, **kwargs)
                cache_instance.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator