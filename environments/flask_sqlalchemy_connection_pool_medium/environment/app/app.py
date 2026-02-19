from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////app/users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

with app.app_context():
    db.create_all()

@app.route('/health', methods=['GET'])
def health():
    # Bug: Creating a new session but never closing it
    session = db.session()
    # Just checking if we can query, but not cleaning up
    session.execute(db.text('SELECT 1'))
    return jsonify({"status": "healthy"})

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    # Bug: Creating a new session manually but never closing it
    session = db.session()
    new_user = User(name=data['name'])
    session.add(new_user)
    session.commit()
    # Missing session.close() - connection remains in pool
    return jsonify({"id": new_user.id, "name": new_user.name})

@app.route('/users', methods=['GET'])
def get_users():
    # Bug: Creating a new scoped session but never removing it
    session = db.session()
    users = session.query(User).all()
    result = [{"id": user.id, "name": user.name} for user in users]
    # Missing session.close() - connection leaks
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)