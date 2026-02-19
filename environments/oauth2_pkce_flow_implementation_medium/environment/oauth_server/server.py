#!/usr/bin/env python3

from flask import Flask, request, jsonify, redirect
import secrets
from datetime import datetime, timedelta
import hashlib
import base64

app = Flask(__name__)

# In-memory storage
authorization_codes = {}
access_tokens = {}

@app.route('/authorize', methods=['GET'])
def authorize():
    """Authorization endpoint - handles authorization requests"""
    client_id = request.args.get('client_id')
    redirect_uri = request.args.get('redirect_uri')
    response_type = request.args.get('response_type')
    state = request.args.get('state')
    scope = request.args.get('scope', '')
    
    # Validate required parameters
    if not all([client_id, redirect_uri, response_type]):
        return jsonify({'error': 'missing_required_parameters'}), 400
    
    if response_type != 'code':
        return jsonify({'error': 'unsupported_response_type'}), 400
    
    # Generate authorization code
    auth_code = secrets.token_urlsafe(32)
    
    # Store authorization code with metadata
    authorization_codes[auth_code] = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'expiry': datetime.now() + timedelta(minutes=10),
        'scope': scope
    }
    
    # Build redirect URL
    redirect_url = f"{redirect_uri}?code={auth_code}"
    if state:
        redirect_url += f"&state={state}"
    
    return redirect(redirect_url)

@app.route('/token', methods=['POST'])
def token():
    """Token endpoint - exchanges authorization code for access token"""
    grant_type = request.form.get('grant_type')
    code = request.form.get('code')
    redirect_uri = request.form.get('redirect_uri')
    client_id = request.form.get('client_id')
    
    # Validate required parameters
    if not all([grant_type, code, redirect_uri, client_id]):
        return jsonify({'error': 'invalid_request'}), 400
    
    if grant_type != 'authorization_code':
        return jsonify({'error': 'unsupported_grant_type'}), 400
    
    # Check if authorization code exists
    if code not in authorization_codes:
        return jsonify({'error': 'invalid_grant'}), 400
    
    code_data = authorization_codes[code]
    
    # Check if code has expired
    if datetime.now() > code_data['expiry']:
        del authorization_codes[code]
        return jsonify({'error': 'invalid_grant'}), 400
    
    # Validate client_id and redirect_uri
    if code_data['client_id'] != client_id:
        return jsonify({'error': 'invalid_client'}), 400
    
    if code_data['redirect_uri'] != redirect_uri:
        return jsonify({'error': 'invalid_grant'}), 400
    
    # Generate access token
    access_token = secrets.token_urlsafe(32)
    access_tokens[access_token] = {
        'client_id': client_id,
        'scope': code_data['scope'],
        'expiry': datetime.now() + timedelta(hours=1)
    }
    
    # Delete used authorization code
    del authorization_codes[code]
    
    # Return access token
    return jsonify({
        'access_token': access_token,
        'token_type': 'Bearer',
        'expires_in': 3600,
        'scope': code_data['scope']
    })

@app.route('/resource', methods=['GET'])
def resource():
    """Protected resource endpoint"""
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        return jsonify({'error': 'missing_authorization'}), 401
    
    # Parse Bearer token
    parts = auth_header.split()
    if len(parts) != 2 or parts[0] != 'Bearer':
        return jsonify({'error': 'invalid_authorization'}), 401
    
    token = parts[1]
    
    # Validate access token
    if token not in access_tokens:
        return jsonify({'error': 'invalid_token'}), 401
    
    token_data = access_tokens[token]
    
    # Check if token has expired
    if datetime.now() > token_data['expiry']:
        del access_tokens[token]
        return jsonify({'error': 'token_expired'}), 401
    
    # Return protected resource
    return jsonify({
        'user_id': '12345',
        'username': 'testuser',
        'email': 'test@example.com',
        'scope': token_data['scope']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)