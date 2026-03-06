import os

# Database configuration for Flask API
# These settings are used to connect to the PostgreSQL database

# Database connection parameters
DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')
DATABASE_PORT = 5433  # BUG: Should be 5432 for PostgreSQL
DATABASE_NAME = os.getenv('DATABASE_NAME', 'apidb')
DATABASE_USER = os.getenv('DATABASE_USER', 'apiuser')
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'default_password')

# Application settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Server configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8080

# Build database connection string
DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"