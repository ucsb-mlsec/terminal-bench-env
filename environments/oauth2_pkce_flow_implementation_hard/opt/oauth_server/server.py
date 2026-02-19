#!/usr/bin/env python3

from flask import Flask, request, jsonify
import hashlib
import base64
import secrets
import string

app = Flask(__name__)

# In-memory storage
authorization_codes = {}
access_tokens = {}

def generate_random_string(length=32):
    """Generate a random string for codes and tokens"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def base64url_encode(data):
    """Base64 URL encode without padding"""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

@app.route('/authorize', methods=['GET'])
def authorize():
    """Authorization endpoint - issues authorization codes"""
    client_id = request.args.get('client_id')
    response_type = request.args.get('response_type')
    code_challenge = request.args.get('code_challenge')
    code_challenge_method = request.args.get('code_challenge_method')
    
    if not all([client_id, response_type, code_challenge, code_challenge_method]):
        return jsonify({'error': 'missing_parameters'}), 400
    
    if response_type != 'code':
        return jsonify({'error': 'unsupported_response_type'}), 400
    
    if code_challenge_method != 'S256':
        return jsonify({'error': 'unsupported_challenge_method'}), 400
    
    # Generate authorization code
    auth_code = generate_random_string(32)
    
    # Store the code challenge
    authorization_codes[auth_code] = {
        'code_challenge': code_challenge,
        'code_challenge_method': code_challenge_method,
        'client_id': client_id
    }
    
    return jsonify({'code': auth_code})

@app.route('/token', methods=['POST'])
def token():
    """Token endpoint - exchanges authorization codes for access tokens"""
    grant_type = request.form.get('grant_type')
    code = request.form.get('code')
    code_verifier = request.form.get('code_verifier')
    
    if not all([grant_type, code, code_verifier]):
        return jsonify({'error': 'missing_parameters'}), 400
    
    if grant_type != 'authorization_code':
        return jsonify({'error': 'unsupported_grant_type'}), 400
    
    # Retrieve stored code challenge
    if code not in authorization_codes:
        return jsonify({'error': 'invalid_code'}), 400
    
    stored_data = authorization_codes[code]
    stored_challenge = stored_data['code_challenge']
    
    # BUG: This is the incorrect comparison - comparing challenge directly to verifier
    # This will always fail because the verifier is not the same as the challenge
    if stored_challenge == code_verifier:
        # Generate access token
        access_token = generate_random_string(64)
        access_tokens[access_token] = {
            'client_id': stored_data['client_id']
        }
        
        # Clean up used authorization code
        del authorization_codes[code]
        
        return jsonify({
            'access_token': access_token,
            'token_type': 'Bearer',
            'expires_in': 3600
        })
    else:
        return jsonify({'error': 'invalid_code_verifier'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)