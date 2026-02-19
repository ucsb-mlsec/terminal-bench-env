#!/usr/bin/env python3

from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'legacy_secret_key_2019'

# MongoDB Configuration
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = 'auth_db'
COLLECTION_NAME = 'users'

# Initialize MongoDB client
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Verify connection
    client.server_info()
    db = client[DATABASE_NAME]
    users_collection = db[COLLECTION_NAME]
    print(f"Successfully connected to MongoDB at {MONGO_URI}")
except ConnectionFailure as e:
    print(f"Failed to connect to MongoDB: {e}")
    db = None
    users_collection = None


@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint - returns application status
    """
    return jsonify({
        "status": "running",
        "application": "Legacy Authentication Service",
        "version": "1.0.0"
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring
    """
    if users_collection is not None:
        return jsonify({"status": "healthy", "database": "connected"}), 200
    else:
        return jsonify({"status": "unhealthy", "database": "disconnected"}), 503


@app.route('/login', methods=['POST'])
def login():
    """
    User authentication endpoint
    Accepts JSON payload with username and password
    Returns success message if credentials are valid
    """
    if users_collection is None:
        return jsonify({"error": "Database connection unavailable"}), 503
    
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Invalid request format"}), 400
    
    # Extract username and password from request
    username = data.get('username')
    password = data.get('password')
    
    if username is None or password is None:
        return jsonify({"error": "Missing username or password"}), 400
    
    # VULNERABLE LINE: User input is directly passed to MongoDB query without sanitization
    # This allows NoSQL injection through MongoDB query operators
    user = users_collection.find_one({"username": username, "password": password})
    
    if user:
        # Authentication successful
        return jsonify({
            "message": "Authentication successful",
            "username": user.get('username'),
            "user_id": str(user.get('_id'))
        }), 200
    else:
        # Authentication failed
        return jsonify({"error": "Invalid credentials"}), 401


@app.route('/register', methods=['POST'])
def register():
    """
    User registration endpoint (for testing purposes)
    """
    if users_collection is None:
        return jsonify({"error": "Database connection unavailable"}), 503
    
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing username or password"}), 400
    
    username = data['username']
    password = data['password']
    
    # Check if user already exists
    existing_user = users_collection.find_one({"username": username})
    if existing_user:
        return jsonify({"error": "User already exists"}), 409
    
    # Insert new user
    users_collection.insert_one({"username": username, "password": password})
    
    return jsonify({"message": "User registered successfully"}), 201


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)