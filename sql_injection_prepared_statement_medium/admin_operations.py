#!/usr/bin/env python3

import sqlite3
import datetime

def list_users_by_role(role):
    """
    List all users with a specific role.
    VULNERABLE: Uses string formatting for SQL query construction.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABLE: String formatting allows SQL injection
    query = "SELECT * FROM users WHERE role = '%s'" % role
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def update_user_status(username, status):
    """
    Update the status of a user.
    VULNERABLE: Uses f-string formatting for SQL query construction.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABLE: f-string formatting allows SQL injection
    query = f"UPDATE users SET status = '{status}' WHERE username = '{username}'"
    cursor.execute(query)
    
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    
    return rows_affected

def get_users_by_date(date_str):
    """
    Get all users created after a specific date.
    VULNERABLE: Uses string concatenation for SQL query construction.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABLE: String concatenation allows SQL injection
    query = "SELECT * FROM users WHERE created_date > '" + date_str + "'"
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def bulk_delete_users(username_list):
    """
    Delete multiple users by username.
    VULNERABLE: Uses f-string formatting with IN clause.
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABLE: f-string with IN clause allows SQL injection
    query = f"DELETE FROM users WHERE username IN ({username_list})"
    cursor.execute(query)
    
    deletion_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deletion_count
