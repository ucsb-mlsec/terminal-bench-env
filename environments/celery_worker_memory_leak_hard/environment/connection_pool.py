#!/usr/bin/env python
# -*- coding: utf-8 -*-

import redis
import threading
import time


class ConnectionPool:
    """Connection pool manager that leaks connections."""
    
    def __init__(self, host='localhost', port=6379, max_connections=10):
        """Initialize the connection pool.
        
        Args:
            host: Redis host
            port: Redis port
            max_connections: Maximum number of connections (not enforced)
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
    
    def get_connection(self):
        """Get a connection from the pool.
        
        Creates a new Redis connection every time and adds it to the list
        without ever removing or reusing existing connections.
        
        Returns:
            redis.Redis: A new Redis connection
        """
        with self.lock:
            conn = redis.Redis(host=self.host, port=self.port, decode_responses=True)
            self.connections.append(conn)
            return conn
    
    def release_connection(self, conn):
        """Release a connection back to the pool.
        
        Does nothing - connections are never closed or returned to pool.
        
        Args:
            conn: The connection to release
        """
        pass
    
    def stats(self):
        """Get pool statistics.
        
        Returns:
            dict: Statistics about the connection pool
        """
        return {'total_connections': len(self.connections)}


# Global connection pool instance
pool = ConnectionPool(host='localhost', port=6379, max_connections=10)


def get_redis_connection():
    """Helper function to get a Redis connection.
    
    Returns:
        redis.Redis: A new Redis connection from the pool
    """
    return pool.get_connection()
