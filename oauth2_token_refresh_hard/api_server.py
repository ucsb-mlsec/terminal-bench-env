#!/usr/bin/env python3

from flask import Flask, request, jsonify
import json
import time
import threading
from datetime import datetime, timedelta
import uuid

app = Flask(__name__)

# In-memory token storage
# Structure: {token_string: {'expires_at': timestamp, 'refresh_token': string}}
tokens = {}
refresh_tokens = {}

def validate_token(token):
    """Validate if token exists and hasn't expired"""
    if token not in tokens:
        return False
    if time.time() > tokens[token]['expires_at']:
        return False
    return True

def generate_customer_data(page, idx):
    """Generate unique customer data"""
    customer_id = f"C{page:02d}{idx:03d}"
    name = f"Customer {customer_id}"
    email = f"customer{customer_id.lower()}@example.com"
    day = ((page - 1) * 15 + idx) % 28 + 1
    created_at = f"2024-01-{day:02d}"
    return {
        'customer_id': customer_id,
        'name': name,
        'email': email,
        'created_at': created_at
    }

@app.route('/oauth/token', methods=['POST'])
def oauth_token():
    """Issue new access token and refresh token"""
    data = request.get_json()
    
    if not data or 'client_id' not in data or 'client_secret' not in data:
        return jsonify({'error': 'invalid_request'}), 400
    
    # Generate tokens
    access_token = str(uuid.uuid4())
    refresh_token = str(uuid.uuid4())
    expires_at = time.time() + 30
    
    # Store tokens
    tokens[access_token] = {
        'expires_at': expires_at,
        'refresh_token': refresh_token
    }
    refresh_tokens[refresh_token] = {
        'access_token': access_token,
        'expires_at': expires_at
    }
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'expires_in': 30
    })

@app.route('/oauth/refresh', methods=['POST'])
def oauth_refresh():
    """Refresh access token using refresh token"""
    data = request.get_json()
    
    if not data or 'refresh_token' not in data:
        return jsonify({'error': 'invalid_request'}), 400
    
    old_refresh_token = data['refresh_token']
    
    # Validate refresh token
    if old_refresh_token not in refresh_tokens:
        return jsonify({'error': 'invalid_token'}), 401
    
    # Generate new tokens
    new_access_token = str(uuid.uuid4())
    new_refresh_token = str(uuid.uuid4())
    expires_at = time.time() + 30
    
    # Invalidate old tokens
    old_access_token = refresh_tokens[old_refresh_token]['access_token']
    if old_access_token in tokens:
        del tokens[old_access_token]
    del refresh_tokens[old_refresh_token]
    
    # Store new tokens
    tokens[new_access_token] = {
        'expires_at': expires_at,
        'refresh_token': new_refresh_token
    }
    refresh_tokens[new_refresh_token] = {
        'access_token': new_access_token,
        'expires_at': expires_at
    }
    
    return jsonify({
        'access_token': new_access_token,
        'refresh_token': new_refresh_token,
        'expires_in': 30
    })

@app.route('/customers', methods=['GET'])
def get_customers():
    """Get paginated customer data"""
    # Check authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'invalid_token'}), 401
    
    token = auth_header.split(' ')[1]
    
    # Validate token
    if not validate_token(token):
        return jsonify({'error': 'invalid_token'}), 401
    
    # Get page parameter
    page = request.args.get('page', type=int)
    if not page or page < 1 or page > 20:
        return jsonify({'error': 'invalid_page'}), 404
    
    # Simulate network delay
    time.sleep(2.5)
    
    # Generate customer data for this page
    customers = []
    for idx in range(1, 16):  # 15 customers per page
        customers.append(generate_customer_data(page, idx))
    
    return jsonify({
        'page': page,
        'customers': customers
    })

if __name__ == '__main__':
    print("Starting API server on port 5000...")
    print("Token expiration: 30 seconds")
    print("Available endpoints:")
    print("  POST /oauth/token")
    print("  POST /oauth/refresh")
    print("  GET /customers?page=N")
    app.run(host='0.0.0.0', port=5000, debug=False)