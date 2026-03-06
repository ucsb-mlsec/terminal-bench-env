#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template_string
import requests
import re
from urllib.parse import urlparse

app = Flask(__name__)

# Simulated metadata storage
METADATA_TOKEN = None
TOKEN_STORE = {}

# HTML form for testing
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>URL Fetcher</title>
</head>
<body>
    <h1>URL Content Fetcher</h1>
    <form method="POST" action="/fetch">
        <label for="url">Enter URL:</label><br>
        <input type="text" id="url" name="url" size="50"><br><br>
        <input type="submit" value="Fetch Content">
    </form>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_FORM)

@app.route('/fetch', methods=['POST'])
def fetch_url():
    if request.is_json:
        data = request.get_json()
        url = data.get('url', '')
    else:
        url = request.form.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    # Basic URL validation with filters (bypassable)
    url_lower = url.lower()
    if 'metadata' in url_lower or '169.254' in url_lower:
        return jsonify({"error": "Blocked: Suspicious URL detected"}), 403
    
    try:
        # Fetch the URL content
        headers = {}
        if 'X-aws-ec2-metadata-token' in request.headers:
            headers['X-aws-ec2-metadata-token'] = request.headers['X-aws-ec2-metadata-token']
        
        response = requests.get(url, timeout=5, headers=headers)
        return jsonify({
            "status": "success",
            "content": response.text,
            "status_code": response.status_code
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simulated AWS Metadata Service endpoints
@app.route('/latest/api/token', methods=['PUT'])
def get_token():
    ttl = request.headers.get('X-aws-ec2-metadata-token-ttl-seconds', '21600')
    token = 'AQAAAHj5K8VxQJGH8mN9pL2kR4tY6uI3oP7qW1eR5tY6uI3oP=='
    TOKEN_STORE[token] = True
    return token, 200

@app.route('/latest/meta-data/instance-id')
def instance_id():
    token = request.headers.get('X-aws-ec2-metadata-token')
    if not token or token not in TOKEN_STORE:
        return "Unauthorized", 401
    return 'i-0a1b2c3d4e5f6g7h8', 200

@app.route('/latest/meta-data/iam/security-credentials/')
def iam_role_list():
    token = request.headers.get('X-aws-ec2-metadata-token')
    if not token or token not in TOKEN_STORE:
        return "Unauthorized", 401
    return 'WebAppRole', 200

@app.route('/latest/meta-data/iam/security-credentials/WebAppRole')
def iam_credentials():
    token = request.headers.get('X-aws-ec2-metadata-token')
    if not token or token not in TOKEN_STORE:
        return "Unauthorized", 401
    
    credentials = {
        "Code": "Success",
        "LastUpdated": "2024-01-15T12:00:00Z",
        "Type": "AWS-HMAC",
        "AccessKeyId": "AKIAIOSFODNN7EXAMPLE",
        "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "Token": "AQoDYXdzEJr1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        "Expiration": "2024-01-15T18:00:00Z"
    }
    return jsonify(credentials), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)