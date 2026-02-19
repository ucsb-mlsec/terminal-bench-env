#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template_string
import secrets
import logging
from datetime import datetime
import hashlib

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simulated user database
users_db = {
    'user@example.com': {
        'password_hash': hashlib.sha256('password123'.encode()).hexdigest(),
        'name': 'John Doe'
    },
    'admin@example.com': {
        'password_hash': hashlib.sha256('admin456'.encode()).hexdigest(),
        'name': 'Admin User'
    }
}

# Store active reset tokens
reset_tokens = {}


@app.route('/')
def home():
    """Home page endpoint"""
    return jsonify({
        'message': 'Welcome to SecureApp',
        'endpoints': ['/login', '/reset-password', '/reset']
    })


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login endpoint"""
    if request.method == 'POST':
        email = request.json.get('email')
        password = request.json.get('password')
        
        if email in users_db:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if users_db[email]['password_hash'] == password_hash:
                logger.info(f"Successful login for {email}")
                return jsonify({'status': 'success', 'message': 'Login successful'})
        
        logger.warning(f"Failed login attempt for {email}")
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
    
    return jsonify({'message': 'Please POST credentials to login'})


@app.route('/reset-password', methods=['POST'])
def reset_password():
    """
    Password reset endpoint - sends reset link to user's email
    This endpoint handles password reset requests by generating a unique token
    and sending a reset link to the user's email address
    """
    email = request.json.get('email') if request.json else None
    
    if not email:
        return jsonify({'status': 'error', 'message': 'Email required'}), 400
    
    if email not in users_db:
        # Still return success to prevent user enumeration
        logger.info(f"Password reset requested for non-existent user: {email}")
        return jsonify({'status': 'success', 'message': 'If account exists, reset link sent'})
    
    # Generate unique reset token
    token = secrets.token_urlsafe(32)
    reset_tokens[token] = {
        'email': email,
        'created_at': datetime.now().isoformat()
    }
    
    # VULNERABLE CODE: Using Host header directly without validation
    # This allows attackers to inject malicious domains in password reset emails
    reset_url = f"https://{request.headers.get('Host')}/reset?token={token}"
    
    # Simulate sending email with reset link
    logger.info(f"Password reset link generated for {email}")
    logger.info(f"Reset URL: {reset_url}")
    print(f"[EMAIL] To: {email}")
    print(f"[EMAIL] Subject: Password Reset Request")
    print(f"[EMAIL] Body: Click here to reset your password: {reset_url}")
    
    return jsonify({
        'status': 'success',
        'message': 'If account exists, reset link sent',
        'debug_token': token  # Remove in production!
    })


@app.route('/reset', methods=['GET', 'POST'])
def reset():
    """Handle password reset with token"""
    if request.method == 'GET':
        token = request.args.get('token')
        if token in reset_tokens:
            return jsonify({
                'status': 'success',
                'message': 'Valid token',
                'email': reset_tokens[token]['email']
            })
        return jsonify({'status': 'error', 'message': 'Invalid or expired token'}), 400
    
    elif request.method == 'POST':
        token = request.json.get('token')
        new_password = request.json.get('new_password')
        
        if token not in reset_tokens:
            return jsonify({'status': 'error', 'message': 'Invalid token'}), 400
        
        email = reset_tokens[token]['email']
        users_db[email]['password_hash'] = hashlib.sha256(new_password.encode()).hexdigest()
        del reset_tokens[token]
        
        logger.info(f"Password successfully reset for {email}")
        return jsonify({'status': 'success', 'message': 'Password updated successfully'})


@app.route('/profile')
def profile():
    """User profile endpoint"""
    return jsonify({
        'message': 'User profile page',
        'note': 'Authentication required'
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    logger.info("Starting SecureApp web application")
    app.run(host='0.0.0.0', port=5000, debug=True)