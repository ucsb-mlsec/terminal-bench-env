#!/usr/bin/env python3

from flask import Flask, request, jsonify
import sqlite3
import json
import os

app = Flask(__name__)

DATABASE = '/home/agent/app.db'

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/user/<username>', methods=['GET'])
def get_user(username):
    """
    GET endpoint to retrieve user information by username.
    PERFORMANCE BOTTLENECK: Must parse JSON preferences on every read.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, username, preferences, created_at FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return jsonify({'error': 'User not found'}), 404
        
        # Parse JSON preferences to extract email - this happens on EVERY read
        try:
            preferences = json.loads(row['preferences'])
            email = preferences.get('email', '')
        except (json.JSONDecodeError, KeyError) as e:
            return jsonify({'error': 'Invalid preferences format'}), 500
        
        return jsonify({
            'id': row['id'],
            'username': row['username'],
            'email': email,
            'preferences': preferences,
            'created_at': row['created_at']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user/<username>/preferences', methods=['POST'])
def update_preferences(username):
    """
    POST endpoint to update user preferences.
    Must read existing JSON, merge updates, and write back as JSON string.
    This is an active write path that will be used during migration.
    """
    try:
        updates = request.get_json()
        if not updates:
            return jsonify({'error': 'No update data provided'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Read existing preferences
        cursor.execute('SELECT preferences FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        
        if row is None:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Parse existing JSON preferences
        try:
            current_preferences = json.loads(row['preferences'])
        except json.JSONDecodeError:
            current_preferences = {}
        
        # Merge updates into existing preferences
        current_preferences.update(updates)
        
        # Write back as JSON string
        updated_json = json.dumps(current_preferences)
        cursor.execute('UPDATE users SET preferences = ? WHERE username = ?', 
                      (updated_json, username))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'preferences': current_preferences
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user/email/<email>', methods=['GET'])
def find_user_by_email(email):
    """
    GET endpoint to find users by email address.
    MAJOR PERFORMANCE BOTTLENECK: Scans entire table and parses JSON for each row!
    With 2,500 records, this is extremely slow and gets worse as data grows.
    No index can help because email is buried in JSON blob.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Must scan ALL rows - no way to index JSON content efficiently in SQLite
        cursor.execute('SELECT id, username, preferences, created_at FROM users')
        rows = cursor.fetchall()
        conn.close()
        
        matching_users = []
        
        # Parse JSON for EVERY SINGLE ROW to find matching email
        # This is O(n) operation that cannot be optimized without schema change
        for row in rows:
            try:
                preferences = json.loads(row['preferences'])
                if preferences.get('email', '').lower() == email.lower():
                    matching_users.append({
                        'id': row['id'],
                        'username': row['username'],
                        'email': preferences['email'],
                        'created_at': row['created_at']
                    })
            except (json.JSONDecodeError, KeyError):
                # Skip malformed JSON
                continue
        
        if not matching_users:
            return jsonify({'error': 'No users found with that email'}), 404
        
        return jsonify({
            'users': matching_users,
            'count': len(matching_users)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Ensure database exists
    if not os.path.exists(DATABASE):
        print(f"ERROR: Database not found at {DATABASE}")
        exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=False)