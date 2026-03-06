#!/usr/bin/env python3

from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)
DATABASE = '/tmp/users.db'

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/login', methods=['POST'])
def login():
    """Authenticate user with username and password"""
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    
    conn = get_db_connection()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    user = conn.execute(query).fetchone()
    conn.close()
    
    if user:
        return jsonify({"status": "success", "user": dict(user)})
    return jsonify({"status": "failed", "message": "Invalid credentials"})

@app.route('/user/search', methods=['GET'])
def search_user():
    """Search for a user by username"""
    username = request.args.get('username', '')
    
    conn = get_db_connection()
    query = "SELECT * FROM users WHERE username='" + username + "'"
    users = conn.execute(query).fetchall()
    conn.close()
    
    return jsonify({"users": [dict(user) for user in users]})

@app.route('/user/profile/<user_id>', methods=['GET'])
def get_profile(user_id):
    """Retrieve user profile by user ID"""
    conn = get_db_connection()
    query = f"SELECT * FROM users WHERE id={user_id}"
    user = conn.execute(query).fetchone()
    conn.close()
    
    if user:
        return jsonify({"profile": dict(user)})
    return jsonify({"error": "User not found"})

@app.route('/user/update', methods=['POST'])
def update_user():
    """Update user email address"""
    user_id = request.form.get('user_id', '')
    email = request.form.get('email', '')
    
    conn = get_db_connection()
    query = "UPDATE users SET email='" + email + "' WHERE id=" + user_id
    conn.execute(query)
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "message": "User updated"})

@app.route('/user/delete', methods=['POST'])
def delete_user():
    """Delete a user account"""
    username = request.form.get('username', '')
    
    conn = get_db_connection()
    query = f"DELETE FROM users WHERE username='{username}'"
    conn.execute(query)
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "message": "User deleted"})

@app.route('/user/role', methods=['GET'])
def get_user_role():
    """Get user role information"""
    role = request.args.get('role', '')
    
    conn = get_db_connection()
    query = "SELECT * FROM users WHERE role='" + role + "'"
    users = conn.execute(query).fetchall()
    conn.close()
    
    return jsonify({"users": [dict(user) for user in users]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)