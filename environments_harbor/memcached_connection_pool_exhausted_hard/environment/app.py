#!/usr/bin/env python3
"""
Production Caching Service using Memcached

This module implements a distributed caching service with connection pooling
to handle high-traffic scenarios. The connection pool size is configurable
and should be tuned based on expected concurrent load.
"""

import logging
import threading
import time
from queue import Queue, Empty, Full
from typing import Any, Optional
from pymemcache.client.base import Client
from pymemcache import exceptions as memcache_exceptions

from config import POOL_SIZE, HOST, PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Thread-safe connection pool for memcached clients.
    
    Manages a pool of memcached connections to avoid connection overhead
    and handle concurrent requests efficiently.
    """
    
    def __init__(self, host: str, port: int, pool_size: int):
        """
        Initialize the connection pool.
        
        Args:
            host: Memcached server host
            port: Memcached server port
            pool_size: Maximum number of connections in the pool
        """
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._initialize_pool()
        
        logger.info(f"Connection pool initialized with size: {pool_size}")
    
    def _initialize_pool(self):
        """Create initial connections and populate the pool."""
        for _ in range(self.pool_size):
            try:
                client = Client((self.host, self.port), connect_timeout=5, timeout=5)
                self.pool.put(client, block=False)
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
    
    def get_connection(self, timeout: float = 5.0) -> Optional[Client]:
        """
        Acquire a connection from the pool.
        
        Args:
            timeout: Maximum time to wait for an available connection
            
        Returns:
            A memcached client connection or None if pool is exhausted
            
        Raises:
            ConnectionError: When no connection is available within timeout
        """
        try:
            connection = self.pool.get(timeout=timeout)
            return connection
        except Empty:
            logger.error("Connection pool exhausted - no connections available")
            raise ConnectionError("Connection pool exhausted")
    
    def return_connection(self, connection: Client):
        """
        Return a connection to the pool.
        
        Args:
            connection: The connection to return to the pool
        """
        try:
            self.pool.put(connection, block=False)
        except Full:
            logger.warning("Connection pool full, closing excess connection")
            connection.close()


class CacheService:
    """
    High-level caching service with connection pool management.
    
    Provides thread-safe cache operations with automatic connection pooling
    to handle concurrent requests efficiently.
    """
    
    def __init__(self, host: str = HOST, port: int = PORT, pool_size: int = POOL_SIZE):
        """
        Initialize the cache service.
        
        Args:
            host: Memcached server host
            port: Memcached server port
            pool_size: Size of the connection pool
        """
        self.pool = ConnectionPool(host, port, pool_size)
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        self.stats_lock = threading.Lock()
        logger.info(f"CacheService initialized (host={host}, port={port}, pool_size={pool_size})")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            The cached value or None if not found or error occurs
        """
        connection = None
        try:
            connection = self.pool.get_connection()
            value = connection.get(key)
            
            with self.stats_lock:
                if value is not None:
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
            
            return value
        except ConnectionError as e:
            logger.error(f"Connection pool exhausted during get operation: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return None
        except memcache_exceptions.MemcacheError as e:
            logger.error(f"Memcache error during get: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return None
        finally:
            if connection:
                self.pool.return_connection(connection)
    
    def set(self, key: str, value: Any, expire: int = 0) -> bool:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to store
            expire: Expiration time in seconds (0 = no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        connection = None
        try:
            connection = self.pool.get_connection()
            result = connection.set(key, value, expire=expire)
            
            with self.stats_lock:
                self.stats['sets'] += 1
            
            return result
        except ConnectionError as e:
            logger.error(f"Connection pool exhausted during set operation: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
        except memcache_exceptions.MemcacheError as e:
            logger.error(f"Memcache error during set: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
        finally:
            if connection:
                self.pool.return_connection(connection)
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        connection = None
        try:
            connection = self.pool.get_connection()
            result = connection.delete(key)
            
            with self.stats_lock:
                self.stats['deletes'] += 1
            
            return result
        except ConnectionError as e:
            logger.error(f"Connection pool exhausted during delete operation: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
        except memcache_exceptions.MemcacheError as e:
            logger.error(f"Memcache error during delete: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
        finally:
            if connection:
                self.pool.return_connection(connection)
    
    def get_stats(self) -> dict:
        """
        Get cache service statistics.
        
        Returns:
            Dictionary containing operation statistics
        """
        with self.stats_lock:
            return self.stats.copy()


# Global cache service instance
cache_service = None


def get_cache_service() -> CacheService:
    """
    Get or create the global cache service instance.
    
    Returns:
        The cache service singleton instance
    """
    global cache_service
    if cache_service is None:
        cache_service = CacheService()
    return cache_service


if __name__ == "__main__":
    # Example usage demonstrating the cache service
    logger.info("Starting cache service example...")
    
    service = get_cache_service()
    
    # Test basic operations
    logger.info("Testing cache operations...")
    
    # Set a value
    success = service.set("test_key", "test_value", expire=300)
    logger.info(f"Set operation: {'success' if success else 'failed'}")
    
    # Get the value
    value = service.get("test_key")
    logger.info(f"Get operation: {value}")
    
    # Delete the value
    success = service.delete("test_key")
    logger.info(f"Delete operation: {'success' if success else 'failed'}")
    
    # Display statistics
    stats = service.get_stats()
    logger.info(f"Service statistics: {stats}")
    
    logger.info("Cache service example completed")