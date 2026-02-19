#!/usr/bin/env python3

import sqlite3

DB_PATH = '/app/data/customers.db'

def get_db_connection():
    """Establish database connection"""
    return sqlite3.connect(DB_PATH)

def authenticate_user(username, password):
    """
    Authenticate user by username and password
    WARNING: VULNERABLE TO SQL INJECTION
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # VULNERABLE: String concatenation
    query = "SELECT * FROM customers WHERE email = '" + username + "' AND account_status = '" + password + "'"
    cursor.execute(query)
    
    result = cursor.fetchone()
    conn.close()
    
    return result

def check_user_exists(email):
    """
    Check if a user exists by email
    WARNING: VULNERABLE TO SQL INJECTION
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # VULNERABLE: f-string formatting
    cursor.execute(f"SELECT COUNT(*) FROM customers WHERE email = '{email}'")
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] > 0 if result else False

def get_user_permissions(user_id, permission_name):
    """
    Get user permissions by user_id and permission name
    WARNING: VULNERABLE TO SQL INJECTION
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # VULNERABLE: %-formatting
    query = "SELECT * FROM customers WHERE id = %s AND account_status = '%s'" % (user_id, permission_name)
    cursor.execute(query)
    
    result = cursor.fetchall()
    conn.close()
    
    return result

def update_last_login(user_id, timestamp):
    """
    Update last login timestamp for user
    WARNING: VULNERABLE TO SQL INJECTION
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # VULNERABLE: .format() method
    query = "UPDATE customers SET account_status = '{}' WHERE id = {}".format(timestamp, user_id)
    cursor.execute(query)
    
    conn.commit()
    conn.close()
    
    return True