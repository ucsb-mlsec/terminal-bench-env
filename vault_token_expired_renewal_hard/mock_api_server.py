#!/usr/bin/env python3

from flask import Flask, request, jsonify

app = Flask(__name__)

# Track token usage counts
token_usage = {}
TOKEN_MAX_USES = 8

def get_token_from_request():
    """Extract token from X-Vault-Token header"""
    return request.headers.get('X-Vault-Token')

def check_token_valid(token):
    """Check if token exists and hasn't exceeded max uses"""
    if not token:
        return False, "missing client token"
    
    if token not in token_usage:
        token_usage[token] = 0
    
    if token_usage[token] >= TOKEN_MAX_USES:
        return False, "permission denied"
    
    return True, None

@app.route('/v1/secret/db-creds', methods=['GET'])
def get_db_creds():
    """Return database credentials if token is valid"""
    token = get_token_from_request()
    valid, error = check_token_valid(token)
    
    if not valid:
        return jsonify({"errors": [error]}), 403
    
    # Increment usage count
    token_usage[token] += 1
    
    return jsonify({
        "data": {
            "username": "dbuser",
            "password": "dbpass123"
        }
    }), 200

@app.route('/v1/auth/token/renew-self', methods=['POST'])
def renew_token():
    """Renew token by resetting its usage count"""
    token = get_token_from_request()
    
    if not token:
        return jsonify({"errors": ["missing client token"]}), 403
    
    # Reset usage count
    token_usage[token] = 0
    
    return jsonify({
        "auth": {
            "client_token": token,
            "renewable": True
        }
    }), 200

@app.route('/v1/auth/token/lookup-self', methods=['GET'])
def lookup_token():
    """Return token information including remaining uses"""
    token = get_token_from_request()
    
    if not token:
        return jsonify({"errors": ["missing client token"]}), 403
    
    if token not in token_usage:
        token_usage[token] = 0
    
    remaining_uses = TOKEN_MAX_USES - token_usage[token]
    
    return jsonify({
        "data": {
            "ttl": remaining_uses
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8200, debug=False)