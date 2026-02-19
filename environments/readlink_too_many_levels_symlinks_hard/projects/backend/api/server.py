#!/usr/bin/env python3
"""
Backend API Server
This is the main server file for the backend API
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    """Main endpoint for the API"""
    return jsonify({
        'status': 'running',
        'message': 'Backend API Server is operational'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'health': 'ok'})

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)