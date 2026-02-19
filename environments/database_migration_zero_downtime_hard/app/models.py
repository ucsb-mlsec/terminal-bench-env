#!/usr/bin/env python3
"""
User management models for the application.

Currently, email addresses are stored inside the JSON preferences blob,
which requires parsing JSON for every email lookup (O(n) performance).
"""

import sqlite3
import json
from typing import Dict, List, Optional


DB_PATH = '/home/agent/app.db'


class User:
    """User model with database operations."""
    
    @staticmethod
    def _get_connection():
        """Helper method to get database connection."""
        return sqlite3.connect(DB_PATH)
    
    @staticmethod
    def get_user_by_username(username: str) -> Optional[Dict]:
        """
        Retrieve user by username.
        
        Args:
            username: The username to look up
            
        Returns:
            Dict with id, username, preferences (parsed JSON as dict), created_at
            or None if not found
        """
        conn = User._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, preferences, created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Parse preferences JSON blob
        preferences = json.loads(row[2]) if row[2] else {}
        
        return {
            'id': row[0],
            'username': row[1],
            'preferences': preferences,
            'created_at': row[3]
        }
    
    @staticmethod
    def get_user_by_email(email: str) -> Optional[Dict]:
        """
        Retrieve user by email address.
        
        WARNING: This is O(n) performance - iterates through all users and
        parses JSON preferences for each one to find matching email.
        This is the performance bottleneck we need to fix.
        
        Args:
            email: The email address to look up
            
        Returns:
            Dict with id, username, preferences (parsed JSON as dict), created_at
            or None if not found
        """
        conn = User._get_connection()
        cursor = conn.cursor()
        
        # O(n) scan - must read all records and parse JSON for each
        cursor.execute("SELECT id, username, preferences, created_at FROM users")
        rows = cursor.fetchall()
        conn.close()
        
        # Iterate through all users parsing JSON to find email match
        for row in rows:
            preferences = json.loads(row[2]) if row[2] else {}
            
            # Email is buried inside the JSON preferences blob
            if preferences.get('email') == email:
                return {
                    'id': row[0],
                    'username': row[1],
                    'preferences': preferences,
                    'created_at': row[3]
                }
        
        return None
    
    @staticmethod
    def update_preferences(username: str, new_prefs_dict: Dict) -> bool:
        """
        Update user preferences.
        
        Converts the preferences dict to a JSON string and stores it in the
        preferences TEXT column.
        
        Args:
            username: Username of the user to update
            new_prefs_dict: New preferences as a dictionary
            
        Returns:
            True if update successful, False otherwise
        """
        conn = User._get_connection()
        cursor = conn.cursor()
        
        # Convert dict to JSON string for storage
        preferences_json = json.dumps(new_prefs_dict)
        
        cursor.execute(
            "UPDATE users SET preferences = ? WHERE username = ?",
            (preferences_json, username)
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    @staticmethod
    def get_all_users() -> List[Dict]:
        """
        Retrieve all users from the database.
        
        Returns:
            List of user dicts with id, username, preferences (parsed JSON), created_at
        """
        conn = User._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, username, preferences, created_at FROM users")
        rows = cursor.fetchall()
        conn.close()
        
        users = []
        for row in rows:
            # Parse the JSON preferences blob for each user
            preferences = json.loads(row[2]) if row[2] else {}
            
            users.append({
                'id': row[0],
                'username': row[1],
                'preferences': preferences,  # Email is inside this JSON object
                'created_at': row[3]
            })
        
        return users