#!/usr/bin/env python3

import os
import sys
import logging
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Read environment variables
APP_ENV = os.getenv('APP_ENV', 'development')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')

# Set log level from environment
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

@app.route('/')
def index():
    return jsonify({"status": "running", "service": "python-api"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/ready')
def ready():
    return jsonify({"status": "ready"}), 200

@app.route('/api/data')
def api_data():
    data = {
        "items": ["item1", "item2", "item3"],
        "count": 3,
        "environment": APP_ENV
    }
    return jsonify(data)

if __name__ == '__main__':
    try:
        logger.info(f"Starting Python API service...")
        logger.info(f"Environment: {APP_ENV}")
        logger.info(f"Log Level: {LOG_LEVEL}")
        logger.info(f"Database Host: {DATABASE_HOST}")
        logger.info(f"Application starting on port 5000")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)