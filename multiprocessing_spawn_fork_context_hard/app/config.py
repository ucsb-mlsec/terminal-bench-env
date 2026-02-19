#!/usr/bin/env python3
"""
Configuration module for the data processing application.
This configuration is designed to be shared across all worker processes.
Global state includes database connections, locks, and shared caches.
"""

import sqlite3
import threading

# Global configuration dictionary
CONFIG_DATA = {'batch_size': 100, 'timeout': 30, 'max_workers': 4}

# Global database connection - shared across processes
DB_CONNECTION = None

# Shared cache for processed results
SHARED_CACHE = {}

# Thread lock for synchronizing access to shared resources
LOCK = None

def init_resources():
    """
    Initialize global resources that should be available to all processes.
    This includes database connections and synchronization primitives.
    """
    global DB_CONNECTION, LOCK
    
    # Create an in-memory database connection
    # This connection object is NOT pickle-serializable
    DB_CONNECTION = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = DB_CONNECTION.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS results (id INTEGER, value TEXT)')
    DB_CONNECTION.commit()
    
    # Initialize threading lock - also not pickle-serializable
    LOCK = threading.Lock()
    
    print("Resources initialized: DB connection and lock created")

# Initialize resources at module import time
# This runs before any multiprocessing starts
init_resources()