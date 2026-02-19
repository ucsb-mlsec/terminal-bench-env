#!/usr/bin/env python3

from flask import Flask, request, jsonify, session
import hashlib
import secrets
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key_12345')

# In-memory storage for demo purposes
user_balances = {
    'USER001': 10000,
    'USER002': 5000,
    'ATK999': 0
}

def generate_csrf_token():
    """Generate a CSRF token for the current session"""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']

def validate_csrf_token(token):
    """Validate the provided CSRF token against the session token"""
    # VULNERABILITY: The validation logic has a critical flaw
    # The condition checks if token is None, but doesn't properly validate empty strings
    # or handle the case when session token doesn't exist
    if token is None:
        return False
    
    session_token = session.get('csrf_token')
    
    # Critical flaw: if session_token doesn't exist, the comparison below
    # will return True when token is an empty string or certain values
    if not session_token:
        # BUG: Should return False here, but instead continues to comparison
        # This allows bypass when no session token exists
        return token == session_token  # None == None is True, "" == None is False
    
    return token == session_token

@app.route('/api/get_token', methods=['GET'])
def get_token():
    """Endpoint to retrieve a valid CSRF token for the current session"""
    token = generate_csrf_token()
    return jsonify({
        'success': True,
        'csrf_token': token
    })

@app.route('/api/transfer', methods=['POST'])
def transfer_money():
    """
    Money transfer endpoint with CSRF protection
    Requires: amount, recipient, token in JSON body
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        amount = data.get('amount')
        recipient = data.get('recipient')
        csrf_token = data.get('token')
        
        # Validate required fields
        if not amount or not recipient:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # CSRF token validation
        # VULNERABILITY: The validate_csrf_token function has a logical flaw
        # When session has no csrf_token, passing None or manipulating the comparison
        # can bypass validation
        if not validate_csrf_token(csrf_token):
            return jsonify({'success': False, 'error': 'Invalid CSRF token'}), 403
        
        # Process the transfer
        sender = 'USER001'  # Simplified - would normally come from authenticated session
        
        if sender not in user_balances or recipient not in user_balances:
            return jsonify({'success': False, 'error': 'Invalid account'}), 400
        
        if user_balances[sender] < amount:
            return jsonify({'success': False, 'error': 'Insufficient funds'}), 400
        
        # Execute transfer
        user_balances[sender] -= amount
        user_balances[recipient] += amount
        
        return jsonify({
            'success': True,
            'message': f'Transferred {amount} to {recipient}',
            'new_balance': user_balances[sender]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/balance/<account_id>', methods=['GET'])
def get_balance(account_id):
    """Get account balance"""
    if account_id in user_balances:
        return jsonify({
            'success': True,
            'account': account_id,
            'balance': user_balances[account_id]
        })
    return jsonify({'success': False, 'error': 'Account not found'}), 404

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'application': 'Money Transfer API',
        'version': '1.0',
        'endpoints': ['/api/get_token', '/api/transfer', '/api/balance/<account_id>']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)