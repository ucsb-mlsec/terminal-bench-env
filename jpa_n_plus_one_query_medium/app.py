from flask import Flask, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Author, Book

app = Flask(__name__)

engine = create_engine('sqlite:///database.db')
Session = sessionmaker(bind=engine)

@app.route('/api/authors', methods=['GET'])
def get_authors():
    session = Session()
    authors = session.query(Author).all()
    
    result = []
    for author in authors:
        result.append({
            'id': author.id,
            'name': author.name,
            'book_count': len(author.books)
        })
    
    session.close()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)