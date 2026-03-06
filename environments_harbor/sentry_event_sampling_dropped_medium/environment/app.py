#!/usr/bin/env python3

import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, request
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# Configure logging
log_dir = '/var/log/app'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add file handler
file_handler = RotatingFileHandler(
    '/var/log/app/application.log',
    maxBytes=10485760,
    backupCount=10
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(console_handler)

# Initialize Sentry SDK with problematic configuration
sentry_sdk.init(
    dsn="https://a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6@o123456.ingest.sentry.io/987654",
    sample_rate=0.1,
    integrations=[FlaskIntegration()],
    environment="production",
    release="api-service@1.2.3",
    traces_sample_rate=1.0,
)

logger.info("Sentry SDK initialized for production environment")

# Initialize Flask application
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# In-memory data store for demo
users_db = {
    '1': {'id': '1', 'name': 'Alice Johnson', 'email': 'alice@example.com'},
    '2': {'id': '2', 'name': 'Bob Smith', 'email': 'bob@example.com'},
    '3': {'id': '3', 'name': 'Charlie Brown', 'email': 'charlie@example.com'},
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancer"""
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'service': 'api-service',
        'version': '1.2.3'
    }), 200

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    logger.info("Fetching all users")
    try:
        return jsonify({
            'success': True,
            'data': list(users_db.values()),
            'count': len(users_db)
        }), 200
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}", exc_info=True)
        raise

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user by ID"""
    logger.info(f"Fetching user with ID: {user_id}")
    try:
        if user_id not in users_db:
            logger.error(f"User not found: {user_id}")
            raise ValueError(f"User with ID {user_id} not found")
        
        return jsonify({
            'success': True,
            'data': users_db[user_id]
        }), 200
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

@app.route('/api/data', methods=['POST'])
def process_data():
    """Process data endpoint - intentionally raises errors for invalid input"""
    logger.info("Processing data request")
    try:
        data = request.get_json()
        
        if not data:
            logger.error("No data provided in request")
            raise ValueError("Request body must contain JSON data")
        
        if 'required_field' not in data:
            logger.error("Missing required_field in request")
            raise KeyError("Missing required field: 'required_field'")
        
        # Simulate processing
        result = {
            'processed': True,
            'input': data,
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        logger.info("Data processed successfully")
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except (ValueError, KeyError) as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error processing data: {str(e)}", exc_info=True)
        raise

@app.route('/api/process', methods=['POST'])
def advanced_process():
    """Advanced processing endpoint with division - can trigger errors"""
    logger.info("Advanced processing request received")
    try:
        data = request.get_json()
        
        numerator = data.get('numerator', 100)
        denominator = data.get('denominator', 1)
        
        # This can cause ZeroDivisionError
        result = numerator / denominator
        
        logger.info(f"Processing completed: {numerator}/{denominator} = {result}")
        return jsonify({
            'success': True,
            'result': result,
            'operation': f'{numerator} / {denominator}'
        }), 200
        
    except ZeroDivisionError as e:
        logger.error("Division by zero error", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in advanced processing: {str(e)}", exc_info=True)
        raise

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(e)
    }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

if __name__ == '__main__':
    logger.info("Starting Flask application on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)