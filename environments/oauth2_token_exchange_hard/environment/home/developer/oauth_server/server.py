#!/usr/bin/env python3

from flask import Flask, request, jsonify
import secrets

app = Flask(__name__)

# Store valid authorization codes
valid_auth_codes = {
    'AUTH_CODE_12345_VALID': {
        'client_id': 'test_client',
        'redirect_uri': 'http://localhost:3000/callback'
    }
}

# BUG #1: Wrong endpoint path - should be /oauth/token
@app.route('/oauth/tokens', methods=['POST'])
def token_endpoint():
    # Validate grant_type
    grant_type = request.form.get('grant_type')
    if grant_type != 'authorization_code':
        return jsonify({'error': 'unsupported_grant_type'}), 400
    
    # Validate client_id
    client_id = request.form.get('client_id')
    if client_id != 'test_client':
        return jsonify({'error': 'invalid_client'}), 400
    
    # BUG #2: Wrong parameter name - should check for 'secret'
    client_secret = request.form.get('client_secret')
    if client_secret != 'test_secret_key':
        return jsonify({'error': 'invalid_client'}), 400
    
    # Validate authorization code
    code = request.form.get('code')
    if code not in valid_auth_codes:
        return jsonify({'error': 'invalid_grant'}), 400
    
    # Validate redirect_uri
    redirect_uri = request.form.get('redirect_uri')
    if redirect_uri != valid_auth_codes[code]['redirect_uri']:
        return jsonify({'error': 'invalid_grant'}), 400
    
    # Validate client_id matches
    if client_id != valid_auth_codes[code]['client_id']:
        return jsonify({'error': 'invalid_grant'}), 400
    
    # Generate access token
    access_token = secrets.token_urlsafe(32)
    
    # BUG #3: Wrong expiration time - should be 3600
    response = {
        'access_token': access_token,
        'token_type': 'Bearer',
        'expires_in': 60
    }
    
    # Remove used authorization code
    del valid_auth_codes[code]
    
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)