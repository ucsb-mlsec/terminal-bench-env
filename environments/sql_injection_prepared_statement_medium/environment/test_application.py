#!/usr/bin/env python3

import unittest
import sqlite3
import os
import sys
import tempfile

# Import the modules to test
import auth_service
import user_profile
import admin_operations


class TestAuthService(unittest.TestCase):
    
    def setUp(self):
        """Set up test database with sample data"""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Initialize database connection
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create schema
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert test users
        test_users = [
            ('alice', 'password123', 'alice@example.com', 'Alice Smith'),
            ('bob', 'securepass', 'bob@example.com', 'Bob Johnson'),
            ('charlie', 'mypassword', 'charlie@example.com', 'Charlie Brown'),
            ('diana', 'pass1234', 'diana@example.com', 'Diana Prince')
        ]
        
        for username, password, email, full_name in test_users:
            self.cursor.execute(
                'INSERT INTO users (username, password, email, full_name) VALUES (?, ?, ?, ?)',
                (username, password, email, full_name)
            )
        
        self.conn.commit()
        
        # Set database path for modules
        auth_service.DB_PATH = self.db_path
        user_profile.DB_PATH = self.db_path
        admin_operations.DB_PATH = self.db_path
    
    def tearDown(self):
        """Clean up test database"""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_login_valid_user(self):
        """Test login with valid credentials"""
        result = auth_service.login('alice', 'password123')
        self.assertTrue(result, "Login should succeed with valid credentials")
    
    def test_login_invalid_user(self):
        """Test login with invalid credentials"""
        result = auth_service.login('alice', 'wrongpassword')
        self.assertFalse(result, "Login should fail with invalid credentials")
    
    def test_register_new_user(self):
        """Test user registration"""
        result = auth_service.register_user('eve', 'evepass', 'eve@example.com', 'Eve Adams')
        self.assertTrue(result, "Registration should succeed for new user")
        
        # Verify user was created
        self.cursor.execute('SELECT username FROM users WHERE username = ?', ('eve',))
        user = self.cursor.fetchone()
        self.assertIsNotNone(user, "New user should exist in database")
        self.assertEqual(user[0], 'eve')
    
    def test_get_user_profile(self):
        """Test retrieving user profile by ID"""
        profile = user_profile.get_profile(1)
        self.assertIsNotNone(profile, "Profile should be retrieved")
        self.assertEqual(profile['username'], 'alice')
        self.assertEqual(profile['email'], 'alice@example.com')
    
    def test_update_email(self):
        """Test email update functionality"""
        result = user_profile.update_email(2, 'bob.new@example.com')
        self.assertTrue(result, "Email update should succeed")
        
        # Verify email was updated
        self.cursor.execute('SELECT email FROM users WHERE id = ?', (2,))
        email = self.cursor.fetchone()
        self.assertEqual(email[0], 'bob.new@example.com')
    
    def test_search_users(self):
        """Test user search functionality"""
        results = admin_operations.search_users('alice')
        self.assertIsNotNone(results, "Search should return results")
        self.assertTrue(len(results) > 0, "Search should find matching users")
        self.assertEqual(results[0]['username'], 'alice')
    
    def test_delete_user(self):
        """Test user deletion"""
        result = admin_operations.delete_user(4)
        self.assertTrue(result, "User deletion should succeed")
        
        # Verify user was deleted
        self.cursor.execute('SELECT * FROM users WHERE id = ?', (4,))
        user = self.cursor.fetchone()
        self.assertIsNone(user, "Deleted user should not exist")


if __name__ == '__main__':
    unittest.main()