from flask import Flask, request, jsonify, make_response
import secrets
import hmac

app = Flask(__name__)

# Simple in-memory user store
users = {"admin": {"password": "admin123", "email": "admin@example.com"}}
sessions = {}
csrf_tokens = {}

def generate_csrf_token():
    """Generate a cryptographically secure CSRF token"""
    return secrets.token_hex(32)

def validate_csrf_token(cookie_token, header_token):
    """Validate CSRF token using constant-time comparison"""
    if not cookie_token or not header_token:
        return False
    return hmac.compare_digest(cookie_token, header_token)

def require_csrf(f):
    """Decorator to require CSRF token validation"""
    def decorated_function(*args, **kwargs):
        # Skip CSRF validation for GET requests
        if request.method == 'GET':
            return f(*args, **kwargs)
        
        # Get tokens from cookie and header
        cookie_token = request.cookies.get('csrf_token')
        header_token = request.headers.get('X-CSRF-Token')
        
        # Validate tokens
        if not validate_csrf_token(cookie_token, header_token):
            return jsonify({"status": "error", "message": "CSRF validation failed"}), 403
        
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    
    if username in users and users[username]['password'] == password:
        session_id = secrets.token_hex(16)
        sessions[session_id] = username
        
        # Generate CSRF token
        csrf_token = generate_csrf_token()
        csrf_tokens[session_id] = csrf_token
        
        resp = make_response(jsonify({
            "status": "success", 
            "message": "Logged in",
            "csrf_token": csrf_token
        }))
        resp.set_cookie('session_id', session_id)
        resp.set_cookie('csrf_token', csrf_token)
        return resp
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route('/update_email', methods=['POST'])
@require_csrf
def update_email():
    session_id = request.cookies.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    data = request.get_json() or {}
    new_email = data.get('email')
    
    username = sessions[session_id]
    users[username]['email'] = new_email
    return jsonify({"status": "success", "message": "Email updated"})

@app.route('/get_profile', methods=['GET'])
def get_profile():
    session_id = request.cookies.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    username = sessions[session_id]
    return jsonify({"username": username, "email": users[username]['email']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)