#!/usr/bin/env python3

import os
import sys
from flask import Flask, jsonify, request
import requests
from config import DATABASE_HOST, DATABASE_PORT, DATABASE_NAME

app = Flask(__name__)

def init_database():
    """Initialize database connection"""
    print(f"Initializing database connection to {DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}")
    
    # Bug #3: Variable referenced before assignment
    if db_connection:
        print("Database connection already exists")
        return True
    
    db_connection = None
    
    try:
        # Simulate database connection setup
        print(f"Connecting to PostgreSQL at {DATABASE_HOST}:{DATABASE_PORT}")
        db_connection = {
            'host': DATABASE_HOST,
            'port': DATABASE_PORT,
            'database': DATABASE_NAME,
            'status': 'connected'
        }
        print("Database connection established successfully")
        return True
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return False

def fetch_external_data():
    """Fetch data from external API using requests library"""
    try:
        # Using requests library to make external API call
        response = requests.get('https://api.example.com/data', timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching external data: {e}")
        return None

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'flask-api',
        'database': 'connected'
    }), 200

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get data from database"""
    try:
        # Simulate database query
        data = {
            'items': [
                {'id': 1, 'name': 'Item 1', 'value': 100},
                {'id': 2, 'name': 'Item 2', 'value': 200},
                {'id': 3, 'name': 'Item 3', 'value': 300}
            ],
            'count': 3
        }
        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['POST'])
def create_data():
    """Create new data entry"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Simulate database insert
        result = {
            'id': 4,
            'name': data.get('name'),
            'value': data.get('value'),
            'status': 'created'
        }
        return jsonify(result), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/external')
def get_external():
    """Fetch and return external data"""
    external_data = fetch_external_data()
    if external_data:
        return jsonify(external_data), 200
    return jsonify({'error': 'Failed to fetch external data'}), 500

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Flask API Service',
        'version': '1.0.0',
        'endpoints': [
            '/health',
            '/api/data',
            '/api/external'
        ]
    }), 200

if __name__ == '__main__':
    print("Starting Flask API service...")
    print(f"Database configuration: {DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}")
    
    # Initialize database connection
    if not init_database():
        print("Failed to initialize database, exiting...")
        sys.exit(1)
    
    print("Starting Flask server on 0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)