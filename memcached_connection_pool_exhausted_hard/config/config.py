#!/usr/bin/env python3
"""
Cache Service Configuration
Configuration file for the distributed caching service using memcached
"""

# Memcached Server Connection Settings
HOST = 'localhost'
PORT = 11211

# Connection Pool Configuration
# Current pool size - insufficient for peak traffic of 600 concurrent connections
POOL_SIZE = 150

# Connection Management
CONNECTION_TIMEOUT = 5  # seconds
SOCKET_TIMEOUT = 3  # seconds
CONNECT_TIMEOUT = 2  # seconds

# Retry and Resilience Settings
RETRY_ATTEMPTS = 3
RETRY_DELAY = 0.5  # seconds
MAX_RETRY_DELAY = 5  # seconds

# Pool Behavior
POOL_MIN_SIZE = 10
POOL_BLOCK = True  # Block when pool is exhausted
POOL_TIMEOUT = 10  # Maximum time to wait for a connection from pool

# Cache Operation Settings
DEFAULT_TTL = 3600  # Default time-to-live in seconds
MAX_KEY_LENGTH = 250
MAX_VALUE_SIZE = 1048576  # 1MB

# Health Check Settings
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 2  # seconds

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance Monitoring
ENABLE_METRICS = True
METRICS_INTERVAL = 60  # seconds