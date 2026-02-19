#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# This application serves user data and includes authentication features
# User data includes profile information and preferences

@app.route('/')
def home():
    """Home page route"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>User Dashboard</title>
    </head>
    <body>
        <h1>Welcome to User Dashboard</h1>
        <p>This application manages user data and profiles.</p>
        <ul>
            <li><a href="/about">About</a></li>
            <li><a href="/api/users">Users API</a></li>
        </ul>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/about')
def about():
    """About page route"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>About</title>
    </head>
    <body>
        <h1>About Our Application</h1>
        <p>This is a user management system that handles sensitive user data.</p>
        <p>Features include user authentication, profile management, and data storage.</p>
        <a href="/">Back to Home</a>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/users')
def get_users():
    """API endpoint that returns user data"""
    users = [
        {"id": 1, "name": "John Doe", "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
        {"id": 3, "name": "Bob Johnson", "email": "bob@example.com"}
    ]
    return jsonify({"users": users, "count": len(users)})

if __name__ == '__main__':
    # Run the Flask application
    # This app serves user data and requires secure access
    app.run(host='0.0.0.0', port=3000, debug=True)