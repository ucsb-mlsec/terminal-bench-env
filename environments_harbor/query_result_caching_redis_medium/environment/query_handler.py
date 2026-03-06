#!/usr/bin/env python3

import redis
import hashlib
import configparser
import sqlite3
import os

# Load Redis configuration
def load_redis_config():
    config = configparser.ConfigParser()
    config_path = '/app/redis_config.ini'
    if os.path.exists(config_path):
        config.read(config_path)
        return config
    return None

# Initialize Redis connection
try:
    config = load_redis_config()
    if config and 'redis' in config:
        redis_client = redis.Redis(
            host=config['redis'].get('host', 'localhost'),
            port=config['redis'].getint('port', 6379),
            db=config['redis'].getint('db', 0),
            decode_responses=True
        )
    else:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

# BROKEN CACHE KEY GENERATION
# BUG: This function only uses the first 15 characters of the query
# This causes cache collisions when multiple queries start with the same text
def generate_cache_key(query):
    """
    Generate a cache key for the given SQL query.
    
    WARNING: This implementation is BROKEN!
    It only considers the first 15 characters of the query,
    which causes different queries with similar beginnings to collide.
    
    For example:
    - "SELECT * FROM users WHERE id = 1"
    - "SELECT * FROM users WHERE id = 2"
    
    Both would generate the same cache key: "query:SELECT * FROM u"
    """
    # BROKEN: Truncating to first 15 characters causes collisions
    truncated = query[:15]
    return f"query:{truncated}"


# CORRECT IMPLEMENTATION (commented out, for reference)
# def generate_cache_key_correct(query):
#     """
#     Correct implementation: Use hash of normalized query.
#     This ensures unique keys for different queries.
#     """
#     # Normalize the query: strip whitespace and convert to lowercase
#     normalized = ' '.join(query.strip().lower().split())
#     # Generate MD5 hash for deterministic, collision-free key
#     query_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
#     return f"query:{query_hash}"


def execute_query_with_cache(query, db_connection=None):
    """
    Execute a database query with Redis caching.
    
    Flow:
    1. Generate cache key from query
    2. Check if result exists in Redis cache
    3. If cached, return cached result
    4. If not cached, execute query against database
    5. Store result in cache and return
    
    Args:
        query: SQL query string to execute
        db_connection: Database connection object (optional)
    
    Returns:
        Query results (list of rows)
    """
    # Generate cache key using the (broken) logic
    cache_key = generate_cache_key(query)
    
    # Try to get cached result
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                print(f"Cache HIT for key: {cache_key}")
                return eval(cached_result)  # Convert string back to list
        except Exception as e:
            print(f"Cache read error: {e}")
    
    # Cache miss - execute query
    print(f"Cache MISS for key: {cache_key}")
    
    # Execute query against database
    results = []
    if db_connection:
        try:
            cursor = db_connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
        except Exception as e:
            print(f"Database query error: {e}")
            return []
    
    # Store result in cache (with 1 hour expiration)
    if redis_client:
        try:
            redis_client.setex(cache_key, 3600, str(results))
        except Exception as e:
            print(f"Cache write error: {e}")
    
    return results


def main():
    """
    Main function to demonstrate query execution with caching.
    """
    # Example usage
    test_query = "SELECT * FROM users WHERE active = 1"
    results = execute_query_with_cache(test_query)
    print(f"Query results: {results}")


if __name__ == "__main__":
    main()