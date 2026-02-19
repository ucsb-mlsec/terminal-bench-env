#!/usr/bin/env python3
"""
Main application entry point
This file changes frequently during development
"""

from flask import Flask, jsonify

# Create Flask application instance
app = Flask(__name__)

# Root route - simple welcome message
@app.route('/')
def home():
    """Main landing page"""
    return 'Hello, World!'

# Health check endpoint for monitoring
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'python-web-app'
    })

# Sample API endpoint
@app.route('/api/data')
def get_data():
    """Return sample data"""
    return jsonify({
        'message': 'Sample data',
        'items': ['item1', 'item2', 'item3'],
        'count': 3
    })

# Application entry point
if __name__ == '__main__':
    # Run the application
    # This code changes often during development
    app.run(host='0.0.0.0', port=8000, debug=True)