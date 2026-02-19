#!/usr/bin/env python3

from flask import Flask, request, jsonify
import re

app = Flask(__name__)

@app.route('/api/session', methods=['POST'])
def handle_session():
    # Log received headers for debugging
    print("=" * 50)
    print("Received headers:")
    for header, value in request.headers:
        print(f"{header}: {value}")
    print("=" * 50)
    
    # Vulnerable header processing - simulates unsafe parsing
    # that allows header injection via newline characters
    
    user_role = 'user'  # Default role
    
    # Check each header for injection attempts
    for header_name, header_value in request.headers:
        # Simulate vulnerable parsing that splits on newlines
        # This mimics how some legacy systems process headers unsafely
        if '\r\n' in header_value or '\n' in header_value:
            # Split the header value on newlines to simulate injection
            injected_lines = re.split(r'\r?\n', header_value)
            for line in injected_lines:
                # Check if an X-User-Role header was injected
                if line.strip().startswith('X-User-Role:'):
                    injected_role = line.split(':', 1)[1].strip()
                    print(f"[VULNERABILITY] Detected injected header: {line}")
                    user_role = injected_role
        
        # Also check the legitimate X-User-Role header
        if header_name == 'X-User-Role':
            user_role = header_value
    
    # Return response based on the role
    if user_role == 'admin':
        response = {"role": "admin", "status": "authenticated"}
        print("[SUCCESS] Admin access granted!")
    else:
        response = {"role": "user", "status": "authenticated"}
        print("[INFO] Regular user access")
    
    return jsonify(response), 200

if __name__ == '__main__':
    print("Starting vulnerable session management service...")
    print("Listening on http://0.0.0.0:8080")
    print("Endpoint: POST /api/session")
    app.run(host='0.0.0.0', port=8080, debug=False)