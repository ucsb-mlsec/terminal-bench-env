#!/usr/bin/env python3
"""
Vulnerable JWT Validator - Educational/Audit Purposes Only
This code demonstrates a path traversal vulnerability in JWT kid parameter handling
"""

from flask import Flask, request, jsonify
import jwt
import json
import os
from datetime import datetime

app = Flask(__name__)

# Base directory for key storage
KEY_STORAGE_DIR = '/opt/vulnerable_app/keys/'

def validate_token(token):
    """
    Vulnerable JWT validation function with path traversal flaw
    WARNING: This function does NOT sanitize the 'kid' parameter!
    """
    try:
        # Decode header without verification to extract kid parameter
        unverified_header = jwt.get_unverified_header(token)
        
        # VULNERABILITY: kid parameter is not sanitized, allowing path traversal
        kid = unverified_header.get('kid', 'default.key')
        
        # CRITICAL FLAW: Directly concatenate user-controlled input to construct file path
        # This allows '../' sequences to escape the KEY_STORAGE_DIR
        key_path = os.path.join(KEY_STORAGE_DIR, kid)
        
        print(f"[DEBUG] Attempting to read key from: {key_path}")
        
        # Read the signing key from the constructed path
        if not os.path.exists(key_path):
            return {"valid": False, "error": "Key file not found"}
        
        with open(key_path, 'r') as key_file:
            signing_key = key_file.read().strip()
        
        # Validate the JWT token using the retrieved key
        decoded_payload = jwt.decode(
            token,
            signing_key,
            algorithms=['HS256']
        )
        
        return {"valid": True, "payload": decoded_payload}
        
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token has expired"}
    except jwt.InvalidTokenError as e:
        return {"valid": False, "error": f"Invalid token: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

@app.route('/validate', methods=['POST'])
def validate():
    """Endpoint to validate JWT tokens"""
    token = request.json.get('token', '')
    
    if not token:
        return jsonify({"error": "No token provided"}), 400
    
    result = validate_token(token)
    
    if result['valid']:
        return jsonify(result), 200
    else:
        return jsonify(result), 401

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)