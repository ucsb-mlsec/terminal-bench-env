from flask import request, jsonify, current_app
from app import db, User

def get_users():
    # BUG: Creates a session but doesn't properly clean it up
    users = User.query.all()
    user_list = []
    for user in users:
        user_list.append({
            'id': user.id,
            'name': user.name,
            'email': user.email
        })
    return jsonify(user_list)

def create_user():
    # BUG: No try-except-finally block, session not rolled back on exception
    data = request.get_json()
    user = User(name=data['name'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email
    }), 201

def get_user(user_id):
    # BUG: Opens a session but never explicitly closes it
    user = User.query.get(user_id)
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email
    })

def register_routes(app):
    app.route('/api/users', methods=['GET'])(get_users)
    app.route('/api/users', methods=['POST'])(create_user)
    app.route('/api/users/<int:user_id>', methods=['GET'])(get_user)