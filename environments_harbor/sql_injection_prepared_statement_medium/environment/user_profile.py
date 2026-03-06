#!/usr/bin/env python3

import sqlite3
import json


def get_user_profile(user_id):
    """
    Get user profile by user ID.
    VULNERABLE: Uses string concatenation for SQL query.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABLE: SQL injection possible through user_id parameter
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    cursor.execute(query)
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'username': result[1],
            'email': result[2],
            'full_name': result[3]
        }
    return None


def update_user_email(username, new_email):
    """
    Update user email by username.
    VULNERABLE: Uses f-string formatting for SQL query.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # VULNERABLE: SQL injection possible through username and new_email parameters
        query = f"UPDATE users SET email = '{new_email}' WHERE username = '{username}'"
        cursor.execute(query)
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except Exception as e:
        conn.close()
        return False


def search_users(search_term):
    """
    Search for users by username pattern.
    VULNERABLE: Uses .format() for SQL query.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABLE: SQL injection possible through search_term parameter
    query = "SELECT username, email FROM users WHERE username LIKE '%{}%'".format(search_term)
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    
    users = []
    for row in results:
        users.append({
            'username': row[0],
            'email': row[1]
        })
    return users


def get_user_by_email(email):
    """
    Get user data by email address.
    VULNERABLE: Uses string concatenation for SQL query.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABLE: SQL injection possible through email parameter
    query = "SELECT * FROM users WHERE email = '" + email + "'"
    cursor.execute(query)
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'username': result[1],
            'email': result[2],
            'full_name': result[3]
        }
    return None