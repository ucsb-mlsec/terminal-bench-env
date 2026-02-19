#!/usr/bin/env python3

from flask import Flask, request, jsonify
import tokens
import db

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    """
    Login endpoint that validates credentials and issues tokens.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    # Validate credentials
    user = db.validate_credentials(username, password)
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    user_id = user['id']
    
    # Generate tokens
    access_token = tokens.generate_access_token(user_id)
    refresh_token = tokens.generate_refresh_token(user_id)
    
    # Store refresh token in database
    db.store_refresh_token(user_id, refresh_token)
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token
    }), 200


@app.route('/refresh', methods=['POST'])
def refresh():
    """
    Token refresh endpoint - VULNERABLE: allows refresh token reuse.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    old_refresh_token = data.get('refresh_token')
    
    if not old_refresh_token:
        return jsonify({'error': 'Refresh token required'}), 400
    
    # Validate the refresh token
    payload = tokens.validate_refresh_token(old_refresh_token)
    
    if not payload:
        return jsonify({'error': 'Invalid refresh token'}), 401
    
    user_id = payload.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'Invalid token payload'}), 401
    
    # VULNERABILITY: Check if token exists but don't invalidate it
    # This allows the same refresh token to be used multiple times
    if not db.is_refresh_token_valid(old_refresh_token):
        return jsonify({'error': 'Refresh token not found or invalid'}), 401
    
    # Generate new tokens
    new_access_token = tokens.generate_access_token(user_id)
    new_refresh_token = tokens.generate_refresh_token(user_id)
    
    # VULNERABILITY: Store new refresh token but don't invalidate the old one
    # This is the critical security flaw - old tokens remain valid
    db.store_refresh_token(user_id, new_refresh_token)
    
    return jsonify({
        'access_token': new_access_token,
        'refresh_token': new_refresh_token
    }), 200


if __name__ == '__main__':
    app.run(debug=True)