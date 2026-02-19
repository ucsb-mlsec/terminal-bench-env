#!/usr/bin/env python3
"""
Web Application Configuration
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-a8f5d2c4b9e1f3a7'
    DEBUG = False
    TESTING = False
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'postgresql://webapp:dbpass123@db:5432/webappdb'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # API Configuration
    API_KEYS = {
        'internal': 'sk_live_a1b2c3d4e5f6g7h8i9j0',
        'external': 'pk_live_x9y8z7w6v5u4t3s2r1q0',
        'stripe': 'sk_live_51HabcdefghijklmnopqrstuvwxyzABCDEFGHI123456',
    }
    
    # Server Configuration
    HOST = '0.0.0.0'
    PORT = 5000
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Session Configuration
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_TIMEOUT = 3600  # seconds
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = '/var/log/webapp/app.log'
    
    # CORS Configuration
    CORS_ALLOWED_ORIGINS = [
        'https://example.com',
        'https://www.example.com',
        'https://api.example.com'
    ]
    CORS_ALLOW_CREDENTIALS = True
    CORS_MAX_AGE = 600

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    SECRET_KEY = 'dev-secret-key-not-for-production'
    DATABASE_URL = 'postgresql://devuser:devpass@localhost:5432/devdb'
    
    HOST = '127.0.0.1'
    PORT = 5000
    
    LOG_LEVEL = 'DEBUG'
    SQLALCHEMY_ECHO = True
    
    SESSION_COOKIE_SECURE = False
    
    CORS_ALLOWED_ORIGINS = ['*']

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'prod-secret-7f8e9d0c1b2a3f4e5d6c'
    
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'postgresql://produser:p@ssw0rd!@prod-db.internal:5432/production'
    
    LOG_LEVEL = 'WARNING'
    LOG_FILE = '/var/log/webapp/production.log'
    
    SESSION_TIMEOUT = 1800  # 30 minutes

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    DATABASE_URL = 'sqlite:///:memory:'
    SECRET_KEY = 'test-secret-key'
    
    LOG_LEVEL = 'DEBUG'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}