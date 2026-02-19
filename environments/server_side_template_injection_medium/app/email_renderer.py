#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'service': 'Email Notification Renderer',
        'version': '1.0'
    })

@app.route('/render', methods=['POST'])
def render_email():
    try:
        data = request.get_json()
        
        # Extract user data
        template = data.get('template', '')
        username = data.get('username', 'User')
        email = data.get('email', 'user@example.com')
        timestamp = data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # VULNERABLE LINE: Directly rendering user-provided template
        rendered = render_template_string(template, username=username, email=email, timestamp=timestamp)
        
        return jsonify({
            'success': True,
            'rendered': rendered
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)