#!/usr/bin/env python3

import sqlite3
import hashlib
import os

DB_PATH = 'users.db'

def init_database():
    """Initialize the database with users table using proper parameterized queries"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # This uses proper parameterized approach for table creation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")


def login_user(username, password):
    """
    VULNERABLE: Login user with SQL injection vulnerability
    Uses string concatenation to build query
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # VULNERABLE: Using f-string concatenation
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
        
        if result:
            print(f"Login successful for user: {username}")
            return True
        else:
            print("Login failed: Invalid credentials")
            return False
    except Exception as e:
        print(f"Error during login: {e}")
        conn.close()
        return False


def register_user(username, password, email):
    """
    VULNERABLE: Register new user with SQL injection vulnerability
    Uses .format() string formatting
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # VULNERABLE: Using .format() string formatting
    query = "INSERT INTO users (username, password, email) VALUES ('{}', '{}', '{}')".format(
        username, password, email
    )
    
    try:
        cursor.execute(query)
        conn.commit()
        conn.close()
        print(f"User {username} registered successfully")
        return "Registration successful"
    except Exception as e:
        print(f"Error during registration: {e}")
        conn.close()
        return f"Registration failed: {e}"


def delete_user(username):
    """
    VULNERABLE: Delete user with SQL injection vulnerability
    Uses string concatenation with + operator
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # VULNERABLE: Using + string concatenation
    query = "DELETE FROM users WHERE username = '" + username + "'"
    
    try:
        cursor.execute(query)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"Deleted {deleted_count} user(s)")
        return deleted_count
    except Exception as e:
        print(f"Error during deletion: {e}")
        conn.close()
        return 0


def get_user_by_email(email):
    """
    VULNERABLE: Get user by email with SQL injection vulnerability
    Uses % string formatting
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # VULNERABLE: Using % string formatting
    query = "SELECT * FROM users WHERE email = '%s'" % email
    
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        print(f"Error fetching user by email: {e}")
        conn.close()
        return None


def update_user_password(username, new_password):
    """
    VULNERABLE: Update user password with SQL injection vulnerability
    Uses f-string concatenation
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # VULNERABLE: Using f-string
    query = f"UPDATE users SET password = '{new_password}' WHERE username = '{username}'"
    
    try:
        cursor.execute(query)
        conn.commit()
        updated_count = cursor.rowcount
        conn.close()
        print(f"Password updated for {username}")
        return updated_count
    except Exception as e:
        print(f"Error updating password: {e}")
        conn.close()
        return 0


if __name__ == "__main__":
    # Initialize database
    init_database()
    
    # Example usage (demonstrations only)
    print("\n--- Testing Registration ---")
    register_user("alice", "password123", "alice@example.com")
    register_user("bob", "securepass", "bob@example.com")
    
    print("\n--- Testing Login ---")
    login_user("alice", "password123")
    login_user("bob", "wrongpass")
    
    print("\n--- Testing Email Lookup ---")
    user = get_user_by_email("alice@example.com")
    if user:
        print(f"Found user: {user}")
    
    print("\n--- Testing Password Update ---")
    update_user_password("alice", "newpassword456")
    
    print("\n--- Testing Deletion ---")
    delete_user("bob")
