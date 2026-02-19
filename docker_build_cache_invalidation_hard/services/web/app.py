#!/usr/bin/env python3
"""
Flask Web Application
Main web service for the application
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.exceptions import HTTPException

# Initialize Flask app
app = Flask(__name__)

# Load configuration from environment variables
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://localhost/myapp')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['DEBUG'] = os.environ.get('DEBUG', 'False').lower() == 'true'

# Initialize database
db = SQLAlchemy(app)

# Error handler for HTTP exceptions
@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors"""
    response = {
        "error": e.name,
        "message": e.description,
        "status": e.code
    }
    return jsonify(response), e.code

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "web",
        "version": "1.0.0"
    })

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users from database"""
    try:
        # Query users from database (example)
        users = [{"id": 1, "name": "John Doe"}, {"id": 2, "name": "Jane Smith"}]
        return jsonify({"users": users, "count": len(users)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user by ID"""
    # Example user lookup
    user = {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}
    return jsonify(user)

@app.route('/api/data', methods=['POST'])
def process_data():
    """Process incoming data"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Process the data (example)
    result = {
        "processed": True,
        "received_keys": list(data.keys()),
        "timestamp": os.environ.get('TIMESTAMP', 'unknown')
    }
    return jsonify(result), 201

if __name__ == '__main__':
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])