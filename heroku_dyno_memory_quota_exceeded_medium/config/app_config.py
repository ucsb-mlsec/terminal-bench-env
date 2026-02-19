#!/usr/bin/env python3
"""
Application configuration for Flask report generation service.
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class for the application."""
    
    # Flask core settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # File upload settings - Very generous limits
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB maximum file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx', 'xls', 'json', 'xml'}
    
    # Cache configuration - No limits set
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 0  # Never expire
    CACHE_SIZE = None  # No size limit - will grow indefinitely
    CACHE_THRESHOLD = 0  # No threshold for cleanup
    
    # Report generation settings
    REPORT_TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'temp_reports')
    REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports')
    MAX_REPORT_SIZE = None  # No limit on report size
    KEEP_INTERMEDIATE_FILES = True  # Keep all temporary files
    
    # Worker configuration
    WORKER_CONNECTIONS = 1000
    WORKER_TIMEOUT = 300  # 5 minutes
    WORKER_CLASS = 'sync'
    
    # Session configuration
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=31)
    
    # Data processing settings
    MAX_ROWS_PER_REPORT = None  # No limit on data processing
    CHUNK_SIZE = None  # Process entire file at once instead of chunks
    ENABLE_COMPRESSION = False
    
    # Memory management - No limits configured
    MAX_MEMORY_PER_REQUEST = None  # No memory limit per request
    ENABLE_GARBAGE_COLLECTION = True
    GC_THRESHOLD = (700, 10, 10)  # Default Python GC thresholds
    
    # Resource cleanup
    AUTO_CLEANUP_UPLOADS = False  # Don't automatically clean up uploads
    AUTO_CLEANUP_TEMP = False  # Don't automatically clean up temp files
    CLEANUP_INTERVAL = None  # No scheduled cleanup
    
    @staticmethod
    def init_app(app):
        """Initialize application with this configuration."""
        # Ensure directories exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.REPORT_TEMP_DIR, exist_ok=True)
        os.makedirs(Config.REPORT_OUTPUT_DIR, exist_ok=True)

class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': Config
}