#!/usr/bin/env python3

from flask import Flask, request, render_template_string
import sqlite3
import secrets
import datetime

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('/var/backups/sessions.db')
    conn.row_factory = sqlite3.Row
    return conn

def generate_token():
    return secrets.token_urlsafe(32)

def send_reset_email(email, reset_link):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('/var/log/mail/smtp.log', 'a') as f:
        f.write(f"[{timestamp}] Sending password reset email to {email} with link: {reset_link}\n")

@app.route('/')
def home():
    return render_template_string('''
        <html>
        <head><title>Secure Web Application</title></head>
        <body>
            <h1>Welcome to Secure Web Application</h1>
            <p><a href="/login">Login</a></p>
            <p><a href="/reset-password">Reset Password</a></p>
        </body>
        </html>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
                           (username, password)).fetchone()
        conn.close()
        
        if user:
            return render_template_string('<h1>Login successful!</h1><p>Welcome, {{ username }}</p>', 
                                        username=username)
        else:
            return render_template_string('<h1>Login failed</h1><p>Invalid credentials</p>')
    
    return render_template_string('''
        <html>
        <head><title>Login</title></head>
        <body>
            <h1>Login</h1>
            <form method="post">
                <label>Username: <input type="text" name="username"></label><br>
                <label>Password: <input type="password" name="password"></label><br>
                <button type="submit">Login</button>
            </form>
            <p><a href="/reset-password">Forgot password?</a></p>
        </body>
        </html>
    ''')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        
        if not username or not email:
            return render_template_string('<h1>Error</h1><p>Username and email required</p>')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND email = ?', 
                           (username, email)).fetchone()
        
        if user:
            token = generate_token()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # VULNERABLE CODE: Using Host header directly without validation
            host = request.headers.get('Host')
            reset_link = f'http://{host}/reset?token={token}'
            
            # Store token in database
            conn.execute('''INSERT INTO password_resets 
                           (username, email, token, reset_link, created_at) 
                           VALUES (?, ?, ?, ?, ?)''',
                        (username, email, token, reset_link, timestamp))
            conn.commit()
            
            # Send email with reset link
            send_reset_email(email, reset_link)
            
            conn.close()
            
            return render_template_string('''
                <h1>Password Reset Requested</h1>
                <p>If your account exists, a password reset link has been sent to your email.</p>
            ''')
        else:
            conn.close()
            return render_template_string('''
                <h1>Password Reset Requested</h1>
                <p>If your account exists, a password reset link has been sent to your email.</p>
            ''')
    
    return render_template_string('''
        <html>
        <head><title>Reset Password</title></head>
        <body>
            <h1>Reset Password</h1>
            <form method="post">
                <label>Username: <input type="text" name="username"></label><br>
                <label>Email: <input type="email" name="email"></label><br>
                <button type="submit">Reset Password</button>
            </form>
            <p><a href="/login">Back to Login</a></p>
        </body>
        </html>
    ''')

@app.route('/reset', methods=['GET'])
def reset_token():
    token = request.args.get('token')
    if not token:
        return render_template_string('<h1>Invalid reset token</h1>')
    
    conn = get_db_connection()
    reset_request = conn.execute('SELECT * FROM password_resets WHERE token = ?', 
                                (token,)).fetchone()
    conn.close()
    
    if reset_request:
        return render_template_string('''
            <h1>Reset Password</h1>
            <form method="post" action="/update-password">
                <input type="hidden" name="token" value="{{ token }}">
                <label>New Password: <input type="password" name="password"></label><br>
                <button type="submit">Update Password</button>
            </form>
        ''', token=token)
    else:
        return render_template_string('<h1>Invalid or expired reset token</h1>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)