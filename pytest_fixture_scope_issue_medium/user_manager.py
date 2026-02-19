#!/usr/bin/env python3

import sqlite3


class UserManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT
            )
        ''')
        self.conn.commit()
    
    def add_user(self, username, email):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO users (username, email) VALUES (?, ?)', (username, email))
        self.conn.commit()
    
    def get_user(self, username):
        cursor = self.conn.cursor()
        cursor.execute('SELECT username, email FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        if row:
            return {'username': row['username'], 'email': row['email']}
        return None
    
    def get_all_users(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT username, email FROM users')
        rows = cursor.fetchall()
        return [{'username': row['username'], 'email': row['email']} for row in rows]
    
    def delete_user(self, username):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM users WHERE username = ?', (username,))
        self.conn.commit()
    
    def close(self):
        self.conn.close()