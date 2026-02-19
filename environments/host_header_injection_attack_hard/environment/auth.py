#!/usr/bin/env python3
# auth.py - Legacy authentication module
# WARNING: This code needs security review before production use

import sqlite3
import datetime
from flask import request
import hashlib
import os

DB_PATH = '/var/backups/sessions.db'

def get_db_connection():
    """Establish database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def validate_credentials(username, password):
    """
    Validate user credentials against database
    Returns True if credentials are valid, False otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Hash the password for comparison
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash)
    )
    user = cursor.fetchone()
    conn.close()
    
    return user is not None

def get_user_by_username(username):
    """
    Retrieve user record by username
    Returns user dict or None
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def get_user_by_email(email):
    """
    Retrieve user record by email
    Returns user dict or None
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def request_password_reset(username_or_email):
    """
    Generate password reset request
    WARNING: This function is vulnerable to Host Header Injection!
    TODO: Implement proper Host header validation
    LEGACY CODE - NEEDS SECURITY REVIEW
    """
    from utils import generate_token
    
    # Look up user by username or email
    user = get_user_by_username(username_or_email)
    if not user:
        user = get_user_by_email(username_or_email)
    
    if not user:
        # Return False but don't reveal if user exists
        return None
    
    # Generate reset token
    token = generate_token()
    
    # VULNERABILITY: Using Host header directly without validation!
    # An attacker can manipulate this header to redirect reset links
    host = request.headers.get('Host', 'localhost')
    
    # Construct reset URL - NO VALIDATION PERFORMED (VULNERABLE!)
    # This allows attacker-controlled domains in password reset links
    reset_link = f"http://{host}/reset-password?token={token}"
    
    # Store token in database with expiration
    conn = get_db_connection()
    cursor = conn.cursor()
    
    expires_at = datetime.datetime.now() + datetime.timedelta(hours=1)
    
    cursor.execute(
        """INSERT INTO password_reset_tokens 
           (user_id, token, reset_link, created_at, expires_at) 
           VALUES (?, ?, ?, ?, ?)""",
        (user['id'], token, reset_link, datetime.datetime.now(), expires_at)
    )
    
    conn.commit()
    conn.close()
    
    # Return the reset link (will be sent via email)
    return reset_link

def check_reset_token_valid(token):
    """
    Check if password reset token is valid and not expired
    Returns user_id if valid, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT user_id, expires_at, used 
           FROM password_reset_tokens 
           WHERE token = ?""",
        (token,)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    token_data = dict(result)
    
    # Check if token has been used
    if token_data['used']:
        return None
    
    # Check if token has expired
    expires_at = datetime.datetime.fromisoformat(token_data['expires_at'])
    if datetime.datetime.now() > expires_at:
        return None
    
    return token_data['user_id']

def process_password_reset(token, new_password):
    """
    Process password reset with valid token
    Returns True if successful, False otherwise
    """
    user_id = check_reset_token_valid(token)
    
    if not user_id:
        return False
    
    # Hash the new password
    password_hash = hashlib.sha256(new_password.encode()).hexdigest()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Update user password
    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id)
    )
    
    # Mark token as used
    cursor.execute(
        "UPDATE password_reset_tokens SET used = 1 WHERE token = ?",
        (token,)
    )
    
    conn.commit()
    conn.close()
    
    return True