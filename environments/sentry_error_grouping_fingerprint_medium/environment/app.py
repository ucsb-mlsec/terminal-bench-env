#!/usr/bin/env python3

import sentry_sdk
from flask import Flask, jsonify, request
import random
import os

# Initialize Sentry SDK without before_send hook (will be added later)
sentry_sdk.init(
    dsn="https://examplePublicKey@o0.ingest.sentry.io/0",
    traces_sample_rate=1.0,
    environment="production"
)

app = Flask(__name__)


class ValidationError(Exception):
    """Custom validation error with rule name"""
    def __init__(self, message, rule_name=None):
        super().__init__(message)
        self.rule_name = rule_name


def simulate_database_connection():
    """Simulate database connection that may timeout"""
    # Randomly raise different timeout error messages
    timeout_messages = [
        'Database connection timeout',
        'Connection to database timed out',
        'DB timeout error',
        'Timeout while connecting to database server',
        'Database server connection timeout exceeded'
    ]
    raise Exception(random.choice(timeout_messages))


def read_config_file(filepath):
    """Simulate reading a configuration file"""
    # Simulate permission denied for reading
    raise PermissionError(f"Permission denied: Cannot read file {filepath}")


def write_log_file(filepath):
    """Simulate writing to a log file"""
    # Simulate permission denied for writing
    raise PermissionError(f"Permission denied: Cannot write to file {filepath}")


def validate_email(email):
    """Validate email format"""
    if '@' not in email:
        raise ValidationError(
            f"Invalid email format: {email}",
            rule_name="email_format"
        )


def validate_age(age):
    """Validate age range"""
    if age < 0 or age > 120:
        raise ValidationError(
            f"Age {age} is out of valid range",
            rule_name="age_range"
        )


def validate_username(username):
    """Validate username length"""
    if len(username) < 3:
        raise ValidationError(
            f"Username {username} is too short",
            rule_name="username_length"
        )


def validate_price(price):
    """Validate price is positive"""
    if price <= 0:
        raise ValidationError(
            f"Price {price} must be positive",
            rule_name="price_positive"
        )


@app.route('/api/users', methods=['POST'])
def create_user():
    """API endpoint to create a user"""
    try:
        data = request.get_json() or {}
        
        # Database connection timeout
        if data.get('trigger') == 'db_timeout':
            simulate_database_connection()
        
        # File read permission error
        if data.get('trigger') == 'file_read':
            read_config_file('/etc/app/users_config.ini')
        
        # Validation errors
        if data.get('trigger') == 'validate_email':
            validate_email(data.get('email', 'invalidemail'))
        
        if data.get('trigger') == 'validate_age':
            validate_age(data.get('age', 150))
        
        if data.get('trigger') == 'validate_username':
            validate_username(data.get('username', 'ab'))
        
        return jsonify({"status": "success", "user_id": 123})
    
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/products', methods=['POST'])
def create_product():
    """API endpoint to create a product"""
    try:
        data = request.get_json() or {}
        
        # Database connection timeout
        if data.get('trigger') == 'db_timeout':
            simulate_database_connection()
        
        # File write permission error
        if data.get('trigger') == 'file_write':
            write_log_file('/var/log/app/products.log')
        
        # Validation error
        if data.get('trigger') == 'validate_price':
            validate_price(data.get('price', -10))
        
        return jsonify({"status": "success", "product_id": 456})
    
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/orders', methods=['POST'])
def create_order():
    """API endpoint to create an order"""
    try:
        data = request.get_json() or {}
        
        # Database connection timeout
        if data.get('trigger') == 'db_timeout':
            simulate_database_connection()
        
        # File read permission error
        if data.get('trigger') == 'file_read':
            read_config_file('/app/config/orders_settings.conf')
        
        # File write permission error
        if data.get('trigger') == 'file_write':
            write_log_file('/var/log/app/orders.log')
        
        return jsonify({"status": "success", "order_id": 789})
    
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/inventory', methods=['GET'])
def get_inventory():
    """API endpoint to get inventory"""
    try:
        trigger = request.args.get('trigger')
        
        # Database connection timeout
        if trigger == 'db_timeout':
            simulate_database_connection()
        
        # File read permission error
        if trigger == 'file_read':
            read_config_file('/data/inventory/stock.csv')
        
        return jsonify({"status": "success", "items": []})
    
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    # Note: In production, the before_send hook should be configured
    # in sentry_sdk.init() to enable custom fingerprinting
    # Example: sentry_sdk.init(dsn="...", before_send=custom_fingerprint)
    
    app.run(debug=True, host='0.0.0.0', port=5000)