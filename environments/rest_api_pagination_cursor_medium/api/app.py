from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/api/posts')
def get_posts():
    conn = sqlite3.connect('/workspace/data/posts.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM posts')
    posts = cursor.fetchall()
    
    posts_list = []
    for post in posts:
        posts_list.append({
            'id': post['id'],
            'title': post['title'],
            'created_at': post['created_at']
        })
    
    conn.close()
    
    return jsonify({'posts': posts_list})

if __name__ == '__main__':
    app.run()