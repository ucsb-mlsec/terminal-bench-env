#!/usr/bin/env python3
"""
Worker module for processing data chunks in parallel.
This module relies on shared global state from config module.
"""

import time
from . import config

def validate_data(data):
    """
    Helper function that validates data using config globals.
    Assumes config module state is already initialized.
    """
    # Check if we've seen this data before in shared cache
    if hasattr(config, 'SHARED_CACHE') and data in config.SHARED_CACHE:
        return config.SHARED_CACHE[data]
    return True

def process_chunk(data):
    """
    Process a chunk of data using shared resources from config module.
    
    This function depends on:
    - config.DB_CONNECTION being already initialized
    - config.SHARED_CACHE for caching results
    - config.LOCK for thread safety
    
    WARNING: These patterns work with 'fork' (copies parent memory) 
    but fail with 'spawn' (fresh interpreter, needs re-initialization)
    """
    # Access global database connection directly
    # This works in fork (inherited) but fails in spawn (not serializable)
    db_conn = config.DB_CONNECTION
    
    # Use shared cache without re-initialization
    # Assumes cache dictionary already exists in parent process
    cache_key = f"data_{hash(str(data))}"
    
    # Attempt to use lock for thread safety
    # Lock objects cannot be pickled for spawn method
    with config.LOCK:
        if cache_key in config.SHARED_CACHE:
            return config.SHARED_CACHE[cache_key]
    
    # Validate using helper that also accesses globals
    if not validate_data(data):
        return None
    
    # Simulate database operation using inherited connection
    result = data * 2 if isinstance(data, (int, float)) else len(str(data))
    
    # Store in shared cache
    with config.LOCK:
        config.SHARED_CACHE[cache_key] = result
    
    return result