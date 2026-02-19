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

def validate_csrf_token(session_id, token_from_header):
    """Validate CSRF token using constant-time comparison"""
    if not session_id or session_id not in csrf_tokens:
        return False
    
    token_from_session = csrf_tokens.get(session_id)
    if not token_from_session or not token_from_header:
        return False
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(token_from_session, token_from_header)

def csrf_protect(f):
    """Decorator to protect routes with CSRF validation"""
    def decorated_function(*args, **kwargs):
        # Only validate CSRF for POST, PUT, DELETE, PATCH methods
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            session_id = request.cookies.get('session_id')
            csrf_token_header = request.headers.get('X-CSRF-Token')
            csrf_token_cookie = request.cookies.get('csrf_token')
            
            # Validate that both cookie and header are present and match
            if not csrf_token_cookie or not csrf_token_header:
                return jsonify({"status": "error", "message": "CSRF token missing"}), 403
            
            # Validate token against session
            if not validate_csrf_token(session_id, csrf_token_header):
                return jsonify({"status": "error", "message": "Invalid CSRF token"}), 403
            
            # Also verify cookie matches header (double submit cookie pattern)
            if not hmac.compare_digest(csrf_token_cookie, csrf_token_header):
                return jsonify({"status": "error", "message": "CSRF token mismatch"}), 403
        
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
@csrf_protect
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