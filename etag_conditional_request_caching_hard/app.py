#!/usr/bin/env python3

from flask import Flask, jsonify, request, make_response
import sqlite3
import hashlib
import json

app = Flask(__name__)
DATABASE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def generate_etag(data):
    """Generate ETag from data by hashing JSON representation"""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()

@app.route('/products', methods=['GET'])
def get_products():
    conn = get_db_connection()
    products = conn.execute('SELECT * FROM products').fetchall()
    conn.close()
    
    products_list = [dict(row) for row in products]
    etag = generate_etag(products_list)
    
    # Check If-None-Match header
    if_none_match = request.headers.get('If-None-Match')
    if if_none_match and if_none_match == etag:
        response = make_response('', 304)
        response.headers['ETag'] = etag
        return response
    
    response = make_response(jsonify(products_list), 200)
    response.headers['ETag'] = etag
    return response

@app.route('/products/<int:id>', methods=['GET'])
def get_product(id):
    conn = get_db_connection()
    product = conn.execute('SELECT * FROM products WHERE id = ?', (id,)).fetchone()
    conn.close()
    
    if product is None:
        return jsonify({'error': 'Product not found'}), 404
    
    product_dict = dict(product)
    etag = generate_etag(product_dict)
    
    # Check If-None-Match header
    if_none_match = request.headers.get('If-None-Match')
    if if_none_match and if_none_match == etag:
        response = make_response('', 304)
        response.headers['ETag'] = etag
        return response
    
    response = make_response(jsonify(product_dict), 200)
    response.headers['ETag'] = etag
    return response

@app.route('/products/<int:id>', methods=['PUT'])
def update_product(id):
    conn = get_db_connection()
    
    # Get current product to generate current ETag
    product = conn.execute('SELECT * FROM products WHERE id = ?', (id,)).fetchone()
    if product is None:
        conn.close()
        return jsonify({'error': 'Product not found'}), 404
    
    current_etag = generate_etag(dict(product))
    
    # Check If-Match header
    if_match = request.headers.get('If-Match')
    if if_match and if_match != current_etag:
        conn.close()
        return jsonify({'error': 'Precondition Failed'}), 412
    
    # Update product
    data = request.get_json()
    update_fields = []
    params = []
    
    if 'name' in data:
        update_fields.append('name = ?')
        params.append(data['name'])
    if 'price' in data:
        update_fields.append('price = ?')
        params.append(data['price'])
    if 'description' in data:
        update_fields.append('description = ?')
        params.append(data['description'])
    
    if update_fields:
        params.append(id)
        query = f"UPDATE products SET {', '.join(update_fields)} WHERE id = ?"
        conn.execute(query, params)
        conn.commit()
    
    # Get updated product
    updated_product = conn.execute('SELECT * FROM products WHERE id = ?', (id,)).fetchone()
    conn.close()
    
    product_dict = dict(updated_product)
    new_etag = generate_etag(product_dict)
    
    response = make_response(jsonify(product_dict), 200)
    response.headers['ETag'] = new_etag
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)