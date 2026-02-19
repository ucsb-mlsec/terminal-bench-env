#!/usr/bin/env python3
# Legacy authentication service - API keys are hardcoded

from flask import Flask, request, jsonify
import sys
import os

# TODO: Move to configuration file
API_KEY = 'legacy_hardcoded_key_12345'

app = Flask(__name__)

@app.route('/validate', methods=['POST'])
def validate():
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({'valid': False, 'message': 'Invalid API key'}), 401
        
        provided_key = data['api_key']
        
        if provided_key == API_KEY:
            return jsonify({'valid': True, 'message': 'Authentication successful'}), 200
        else:
            return jsonify({'valid': False, 'message': 'Invalid API key'}), 401
    except Exception as e:
        return jsonify({'valid': False, 'message': 'Invalid API key'}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)