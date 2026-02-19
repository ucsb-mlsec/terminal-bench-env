#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template_string
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('/opt/webapp/database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>User Lookup Application</title>
    </head>
    <body>
        <h1>User Lookup System</h1>
        <p>This is a vulnerable web application for security testing.</p>
        <h2>User Lookup</h2>
        <form action="/user" method="GET">
            <label for="username">Username:</label>
            <input type="text" id="username" name="username" required>
            <button type="submit">Search</button>
        </form>
        <p>Example: Try searching for "admin"</p>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/user', methods=['GET', 'POST'])
def lookup_user():
    if request.method == 'POST':
        username = request.form.get('username', '')
    else:
        username = request.args.get('username', '')
    
    if not username:
        return jsonify({"error": "No username provided"}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Vulnerable SQL query - direct string concatenation
        query = f"SELECT * FROM users WHERE username = '{username}'"
        cursor.execute(query)
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            users = []
            for row in results:
                user_dict = dict(row)
                users.append(user_dict)
            return jsonify({"users": users})
        else:
            return jsonify({"message": "No users found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/lookup', methods=['GET', 'POST'])
def lookup_by_id():
    if request.method == 'POST':
        user_id = request.form.get('id', '')
    else:
        user_id = request.args.get('id', '')
    
    if not user_id:
        return jsonify({"error": "No ID provided"}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Vulnerable SQL query - direct string concatenation
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor.execute(query)
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            users = []
            for row in results:
                user_dict = dict(row)
                users.append(user_dict)
            return jsonify({"users": users})
        else:
            return jsonify({"message": "No users found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)