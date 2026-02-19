#!/usr/bin/env python3

from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Valid credentials and authorization code
VALID_CLIENT_ID = 'test_client_123'
VALID_CLIENT_SECRET = 'secret_key_xyz_789'
VALID_AUTH_CODE = 'AUTH_CODE_abc123def456'
VALID_GRANT_TYPE = 'authorization_code'

# Track used authorization codes (codes can only be used once)
used_codes = set()

@app.route('/token', methods=['POST'])
def token_endpoint():
    """OAuth2 token endpoint that exchanges authorization code for access token"""
    
    # Get parameters from request (supports both form data and JSON)
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
    
    grant_type = data.get('grant_type')
    code = data.get('code')
    client_id = data.get('client_id')
    client_secret = data.get('client_secret')
    
    logging.info(f"Token request received - grant_type: {grant_type}, client_id: {client_id}")
    
    # Validate grant_type
    if grant_type != VALID_GRANT_TYPE:
        logging.warning(f"Invalid grant_type: {grant_type}")
        return jsonify({
            'error': 'unsupported_grant_type',
            'error_description': 'The grant type is not supported'
        }), 400
    
    # Validate client credentials
    if client_id != VALID_CLIENT_ID:
        logging.warning(f"Invalid client_id: {client_id}")
        return jsonify({
            'error': 'invalid_client',
            'error_description': 'Invalid client credentials'
        }), 401
    
    if client_secret != VALID_CLIENT_SECRET:
        logging.warning(f"Invalid client_secret")
        return jsonify({
            'error': 'invalid_client',
            'error_description': 'Invalid client credentials'
        }), 401
    
    # Validate authorization code
    if code != VALID_AUTH_CODE:
        logging.warning(f"Invalid authorization code: {code}")
        return jsonify({
            'error': 'invalid_grant',
            'error_description': 'Invalid authorization code'
        }), 400
    
    # Check if code has already been used
    if code in used_codes:
        logging.warning(f"Authorization code already used: {code}")
        return jsonify({
            'error': 'invalid_grant',
            'error_description': 'Authorization code has already been used'
        }), 400
    
    # Mark code as used
    used_codes.add(code)
    
    # Generate successful token response
    logging.info("Token exchange successful")
    return jsonify({
        'access_token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlRlc3QgVXNlciIsImlhdCI6MTUxNjIzOTAyMn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c',
        'token_type': 'Bearer',
        'expires_in': 3600,
        'refresh_token': 'refresh_token_xyz789abc123'
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    logging.info("Starting OAuth2 mock server on port 9090...")
    app.run(host='0.0.0.0', port=9090, debug=False)