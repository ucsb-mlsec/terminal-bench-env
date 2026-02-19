#!/usr/bin/env python3

from flask import Flask, request, make_response, redirect, render_template_string
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Web Application - Endpoints</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        ul { line-height: 2; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Web Application Endpoints</h1>
    <div class="endpoint">
        <strong>1. Language Selector:</strong> <a href="/set_language?lang=en">/set_language?lang=en</a>
        <p>Set your preferred language via query parameter</p>
    </div>
    <div class="endpoint">
        <strong>2. URL Redirector:</strong> <a href="/redirect?url=https://example.com">/redirect?url=https://example.com</a>
        <p>Redirect to external URLs</p>
    </div>
    <div class="endpoint">
        <strong>3. User Preferences:</strong> <a href="/set_preferences?theme=dark&timezone=UTC">/set_preferences?theme=dark&timezone=UTC</a>
        <p>Save user preferences in cookies</p>
    </div>
    <div class="endpoint">
        <strong>4. Custom Headers API:</strong> <a href="/api/custom_headers">/api/custom_headers</a>
        <p>POST endpoint for adding custom response headers (send form data)</p>
    </div>
    <div class="endpoint">
        <strong>5. Analytics Tracker:</strong> <a href="/track">/track</a>
        <p>Track page visits using Referer header</p>
    </div>
    <div class="endpoint">
        <strong>6. Profile Update:</strong> <a href="/update_profile?username=john&email=john@example.com">/update_profile?username=john&email=john@example.com</a>
        <p>Update user profile information</p>
    </div>
    <div class="endpoint">
        <strong>7. Cache Control:</strong> <a href="/content?cache_duration=3600&content_type=application/json">/content?cache_duration=3600&content_type=application/json</a>
        <p>Serve content with custom cache settings</p>
    </div>
    <div class="endpoint">
        <strong>8. Debug Info:</strong> <a href="/debug?client_id=12345">/debug?client_id=12345</a>
        <p>Debug endpoint for troubleshooting</p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return HOME_PAGE

@app.route('/set_language')
def set_language():
    lang = request.args.get('lang', 'en')
    response = make_response(f"Language set to: {lang}")
    response.headers['X-Language'] = lang
    response.headers['Content-Language'] = lang
    return response

@app.route('/redirect')
def redirect_url():
    target_url = request.args.get('url', '/')
    return redirect(target_url)

@app.route('/set_preferences')
def set_preferences():
    theme = request.args.get('theme', 'light')
    timezone = request.args.get('timezone', 'UTC')
    
    response = make_response("Preferences saved successfully")
    response.set_cookie('user_theme', theme)
    response.set_cookie('user_timezone', timezone)
    response.headers['X-Theme-Set'] = theme
    return response

@app.route('/api/custom_headers', methods=['GET', 'POST'])
def custom_headers():
    if request.method == 'POST':
        header_name = request.form.get('header_name', 'X-Custom')
        header_value = request.form.get('header_value', 'default')
        
        response = make_response({"status": "success", "message": "Headers set"})
        response.headers[header_name] = header_value
        response.headers['X-Request-ID'] = request.form.get('request_id', 'none')
        return response
    else:
        return """
        <html>
        <body>
            <h2>Custom Headers API</h2>
            <form method="POST">
                Header Name: <input type="text" name="header_name" value="X-Custom"><br><br>
                Header Value: <input type="text" name="header_value"><br><br>
                Request ID: <input type="text" name="request_id"><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
        </html>
        """

@app.route('/track')
def track():
    referer = request.headers.get('Referer', 'direct')
    user_agent = request.headers.get('User-Agent', 'unknown')
    
    response = make_response("Tracking pixel loaded")
    response.headers['X-Referer-Source'] = referer
    response.headers['X-Tracked-Agent'] = user_agent
    response.headers['X-Tracking-ID'] = request.args.get('tid', 'none')
    return response

@app.route('/update_profile')
def update_profile():
    username = request.args.get('username', '')
    email = request.args.get('email', '')
    display_name = request.args.get('display_name', username)
    
    response = make_response(f"Profile updated for {username}")
    response.headers['X-Username'] = username
    response.headers['X-User-Email'] = email
    response.headers['X-Display-Name'] = display_name
    response.headers['X-Profile-Updated'] = 'true'
    
    return response

@app.route('/content')
def content():
    cache_duration = request.args.get('cache_duration', '0')
    content_type = request.args.get('content_type', 'text/plain')
    filename = request.args.get('filename', 'document.txt')
    
    response = make_response("Content served successfully")
    response.headers['Cache-Control'] = f'max-age={cache_duration}'
    response.headers['Content-Type'] = content_type
    response.headers['Content-Disposition'] = f'inline; filename={filename}'
    response.headers['X-Content-Source'] = request.args.get('source', 'server')
    
    return response

@app.route('/debug')
def debug():
    client_id = request.args.get('client_id', 'unknown')
    session_id = request.args.get('session_id', 'none')
    debug_mode = request.args.get('debug_mode', 'off')
    
    response = make_response(f"""
    <html>
    <body>
        <h2>Debug Information</h2>
        <p>Client ID: {client_id}</p>
        <p>Session ID: {session_id}</p>
        <p>Debug Mode: {debug_mode}</p>
    </body>
    </html>
    """)
    
    response.headers['X-Debug-Client'] = client_id
    response.headers['X-Debug-Session'] = session_id
    response.headers['X-Debug-Mode'] = debug_mode
    response.headers['X-Debug-Timestamp'] = request.args.get('timestamp', 'not-set')
    
    return response

@app.route('/api/log', methods=['POST'])
def log_api():
    log_level = request.form.get('level', 'INFO')
    log_category = request.form.get('category', 'general')
    
    response = make_response({"status": "logged"})
    response.headers['X-Log-Level'] = log_level
    response.headers['X-Log-Category'] = log_category
    
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)