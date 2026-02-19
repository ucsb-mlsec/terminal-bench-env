#!/usr/bin/env python3
from flask import Flask, request, jsonify

app = Flask(__name__)

# Simple in-memory user database
users = {
    'admin': {'password': 'admin123'},
    'user1': {'password': 'password1'},
    'user2': {'password': 'password2'}
}

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the User Authentication Service'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password required'}), 400
    
    username = data['username']
    password = data['password']
    
    if username in users and users[username]['password'] == password:
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password required'}), 400
    
    username = data['username']
    password = data['password']
    
    if username in users:
        return jsonify({'error': 'Username already exists'}), 409
    
    users[username] = {'password': password}
    return jsonify({'success': True, 'message': 'User registered successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)