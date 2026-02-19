#!/usr/bin/env python3

from flask import Flask, request, render_template_string, render_template, redirect, url_for
import os
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev_secret_key_12345'

# Store some sample data
users_database = {
    'admin': {'email': 'admin@example.com', 'role': 'administrator'},
    'user1': {'email': 'user1@example.com', 'role': 'user'},
    'user2': {'email': 'user2@example.com', 'role': 'user'}
}

visits = []

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Welcome to UserGreet Portal</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #333; }
            a { color: #007bff; text-decoration: none; margin: 10px; display: inline-block; }
            a:hover { text-decoration: underline; }
            .info { background: #e7f3ff; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to UserGreet Portal</h1>
            <p>A simple greeting service for personalized messages.</p>
            
            <div class="info">
                <h3>Available Services:</h3>
                <a href="/greet?name=Guest">Personalized Greeting</a><br>
                <a href="/about">About Us</a><br>
                <a href="/stats">Visit Statistics</a>
            </div>
            
            <p>Version 2.1.3 - Last Updated: 2024-01-15</p>
        </div>
    </body>
    </html>
    '''
    return html

@app.route('/greet')
def greet():
    name = request.args.get('name', 'Guest')
    
    # Log the visit
    visits.append({
        'name': name,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ip': request.remote_addr
    })
    
    # Create personalized greeting
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Greetings</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #28a745; }
            .greeting { font-size: 24px; margin: 20px 0; }
            a { color: #007bff; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome!</h1>
            <div class="greeting">
                Hello, ''' + name + '''!
            </div>
            <p>We're glad to have you here.</p>
            <a href="/">Back to Home</a>
        </div>
    </body>
    </html>
    '''
    
    return render_template_string(template)

@app.route('/about')
def about():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>About Us</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #333; }
            a { color: #007bff; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>About UserGreet Portal</h1>
            <p>UserGreet Portal is a demonstration application that provides personalized greeting services.</p>
            <p>Built with Flask and modern web technologies.</p>
            <h3>Features:</h3>
            <ul>
                <li>Personalized greetings</li>
                <li>Visit tracking</li>
                <li>User-friendly interface</li>
            </ul>
            <p><a href="/">Back to Home</a></p>
        </div>
    </body>
    </html>
    '''
    return html

@app.route('/stats')
def stats():
    total_visits = len(visits)
    recent_visits = visits[-5:] if len(visits) > 5 else visits
    
    visits_html = ''
    for visit in reversed(recent_visits):
        visits_html += f"<li>{visit['name']} - {visit['timestamp']} from {visit['ip']}</li>"
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Statistics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #333; }}
            a {{ color: #007bff; text-decoration: none; }}
            ul {{ background: #f9f9f9; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Visit Statistics</h1>
            <p>Total visits: {total_visits}</p>
            <h3>Recent Visits:</h3>
            <ul>
                {visits_html if visits_html else '<li>No visits yet</li>'}
            </ul>
            <p><a href="/">Back to Home</a></p>
        </div>
    </body>
    </html>
    '''
    return html

@app.route('/users')
def users():
    user_list = ''
    for username, data in users_database.items():
        user_list += f"<li><strong>{username}</strong> - {data['email']} ({data['role']})</li>"
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Users</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #333; }}
            a {{ color: #007bff; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Registered Users</h1>
            <ul>
                {user_list}
            </ul>
            <p><a href="/">Back to Home</a></p>
        </div>
    </body>
    </html>
    '''
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)