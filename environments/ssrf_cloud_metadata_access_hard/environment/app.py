#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template_string
import requests
from urllib.parse import urlparse
import socket

app = Flask(__name__)

# Simulated AWS metadata storage
METADATA = {
    'instance-id': 'i-0a1b2c3d4e5f6g7h8',
    'iam-role': 'EC2-WebServer-Role',
    'credentials': {
        'AccessKeyId': 'AKIAIOSFODNN7EXAMPLE',
        'SecretAccessKey': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
        'Token': 'FwoGZXIvYXdzEBYaDJqXvQk8RjvZvXxLRyLAAT8K1QcOJaLqZzQxNyU7J+cZ8ksB7g9eHf3CjTMPL9J1vB2KzQ8bxLaR4XnYjDfGhTzQp8VtM3Lk7RnF9XvJpK4NmQ2W8sY5HcT6fE1GwJ3UxR7PzL9hV8kNqT4m',
        'Expiration': '2024-12-31T23:59:59Z'
    }
}

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>URL Fetcher Service</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        .endpoint { background: #f4f4f4; padding: 15px; margin: 20px 0; border-radius: 5px; }
        code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>URL Fetcher Service</h1>
        <p>Welcome to our URL fetching service! This tool allows you to fetch content from remote URLs.</p>
        
        <div class="endpoint">
            <h2>Usage</h2>
            <p><strong>Endpoint:</strong> <code>/fetch</code></p>
            <p><strong>Parameter:</strong> <code>url</code> (the URL to fetch)</p>
            <p><strong>Example:</strong></p>
            <code>curl "http://localhost:5000/fetch?url=https://example.com"</code>
        </div>
        
        <div class="endpoint">
            <h2>Features</h2>
            <ul>
                <li>Fetch content from any public URL</li>
                <li>Returns the raw response content</li>
                <li>Built-in security filtering to prevent abuse</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h2>Security</h2>
            <p>This service implements URL filtering to prevent access to internal services and cloud metadata endpoints.</p>
        </div>
    </div>
</body>
</html>
"""

def is_metadata_request(url):
    """Check if the URL is trying to access metadata service - basic filtering"""
    url_lower = url.lower()
    # Block obvious attempts
    if '169.254.169.254' in url_lower:
        return True
    if 'metadata' in url_lower:
        return True
    return False

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/fetch')
def fetch_url():
    url = request.args.get('url', '')
    
    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400
    
    # Basic validation - can be bypassed!
    if is_metadata_request(url):
        return jsonify({'error': 'Access to metadata service is blocked'}), 403
    
    try:
        # Vulnerable SSRF - makes request to user-supplied URL
        response = requests.get(url, timeout=5)
        return response.text, response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

# Mock AWS Metadata Service endpoints
@app.route('/latest/meta-data/')
def metadata_root():
    """Simulate metadata service root directory"""
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        directory_listing = """instance-id
iam/
ami-id
hostname
local-ipv4
public-ipv4"""
        return directory_listing, 200
    return "Not Found", 404

@app.route('/latest/meta-data/instance-id')
def metadata_instance_id():
    """Return instance ID"""
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        return METADATA['instance-id'], 200
    return "Not Found", 404

@app.route('/latest/meta-data/iam/')
def metadata_iam_root():
    """IAM directory listing"""
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        return "security-credentials/", 200
    return "Not Found", 404

@app.route('/latest/meta-data/iam/security-credentials/')
def metadata_iam_roles():
    """Return IAM role name"""
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        return METADATA['iam-role'], 200
    return "Not Found", 404

@app.route('/latest/meta-data/iam/security-credentials/<role_name>')
def metadata_iam_credentials(role_name):
    """Return IAM credentials for the role"""
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        if role_name == METADATA['iam-role']:
            creds = {
                'Code': 'Success',
                'LastUpdated': '2024-01-15T12:00:00Z',
                'Type': 'AWS-HMAC',
                'AccessKeyId': METADATA['credentials']['AccessKeyId'],
                'SecretAccessKey': METADATA['credentials']['SecretAccessKey'],
                'Token': METADATA['credentials']['Token'],
                'Expiration': METADATA['credentials']['Expiration']
            }
            return jsonify(creds), 200
    return "Not Found", 404

# Additional metadata endpoints
@app.route('/latest/meta-data/ami-id')
def metadata_ami_id():
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        return "ami-0c55b159cbfafe1f0", 200
    return "Not Found", 404

@app.route('/latest/meta-data/hostname')
def metadata_hostname():
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        return "ip-172-31-45-123.ec2.internal", 200
    return "Not Found", 404

@app.route('/latest/meta-data/local-ipv4')
def metadata_local_ip():
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        return "172.31.45.123", 200
    return "Not Found", 404

@app.route('/latest/meta-data/public-ipv4')
def metadata_public_ip():
    if request.remote_addr == '127.0.0.1' or request.headers.get('X-Forwarded-For'):
        return "54.123.45.67", 200
    return "Not Found", 404

if __name__ == '__main__':
    print("Starting vulnerable Flask application...")
    print("URL Fetcher Service running on http://localhost:5000")
    print("Vulnerable endpoint: /fetch?url=<target_url>")
    app.run(host='0.0.0.0', port=5000, debug=False)