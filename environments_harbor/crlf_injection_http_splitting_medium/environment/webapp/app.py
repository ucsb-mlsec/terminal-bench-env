#!/usr/bin/env python3

from flask import Flask, request, make_response, redirect, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    """Safe homepage route"""
    html = """
    <html>
        <head><title>Welcome to SecureApp</title></head>
        <body>
            <h1>Welcome to SecureApp</h1>
            <p>A simple web application for managing user preferences.</p>
            <ul>
                <li><a href="/download?filename=report.pdf">Download Report</a></li>
                <li><a href="/setlang?lang=en">Set Language</a></li>
                <li><a href="/profile">View Profile</a></li>
            </ul>
        </body>
    </html>
    """
    return html

@app.route('/download')
def download():
    """Vulnerable function that sets Content-Disposition header with user input"""
    filename = request.args.get('filename', 'document.pdf')
    
    response = make_response("File content would be here")
    # Vulnerable: directly inserting user input into header without sanitization
    response.headers['Content-Disposition'] = 'attachment; filename=' + filename
    response.headers['Content-Type'] = 'application/pdf'
    
    return response

@app.route('/setlang')
def setlang():
    """Vulnerable function that sets custom language header with user input"""
    lang = request.args.get('lang', 'en')
    
    response = make_response("Language preference updated")
    # Vulnerable: directly inserting user input into header without sanitization
    response.headers['X-Language-Preference'] = lang
    response.headers['Content-Type'] = 'text/plain'
    
    return response

@app.route('/profile')
def profile():
    """Safe profile route"""
    html = """
    <html>
        <head><title>User Profile</title></head>
        <body>
            <h1>User Profile</h1>
            <p>Username: demo_user</p>
            <p>Email: demo@example.com</p>
            <p>Role: Standard User</p>
        </body>
    </html>
    """
    return html

@app.route('/redirect')
def redirect_user():
    """Safe redirect function"""
    target = request.args.get('url', '/')
    # Using Flask's built-in redirect which is safe
    if target.startswith('/'):
        return redirect(target)
    return redirect('/')

@app.route('/api/status')
def api_status():
    """Safe API endpoint"""
    response = make_response('{"status": "ok", "version": "1.0"}')
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)