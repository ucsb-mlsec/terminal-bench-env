#!/usr/bin/env python3

from flask import Flask, request, jsonify
import sqlite3
import base64
import json

app = Flask(__name__)

DATABASE = '/home/project/products.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def encode_cursor(created_at, id):
    """Encode the cursor from created_at and id"""
    cursor_data = {'created_at': created_at, 'id': id}
    cursor_json = json.dumps(cursor_data)
    return base64.b64encode(cursor_json.encode()).decode()

def decode_cursor(cursor):
    """Decode the cursor to get created_at and id"""
    try:
        cursor_json = base64.b64decode(cursor.encode()).decode()
        return json.loads(cursor_json)
    except:
        return None

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if cursor-based or offset-based pagination
        cursor_param = request.args.get('cursor')
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        
        if cursor_param:
            # Cursor-based pagination
            cursor_data = decode_cursor(cursor_param)
            if cursor_data is None:
                return jsonify({'error': 'Invalid cursor'}), 400
            
            created_at = cursor_data['created_at']
            id = cursor_data['id']
            
            query = '''
                SELECT * FROM products 
                WHERE (created_at > ? OR (created_at = ? AND id > ?))
                ORDER BY created_at, id 
                LIMIT ?
            '''
            cursor.execute(query, (created_at, created_at, id, limit))
            products = cursor.fetchall()
            
            # Build response
            product_list = []
            next_cursor = None
            
            for product in products:
                product_dict = {
                    'id': product['id'],
                    'name': product['name'],
                    'price': product['price'],
                    'created_at': product['created_at'],
                    'category': product['category']
                }
                product_list.append(product_dict)
            
            # Generate next cursor if there are results
            if product_list:
                last_product = products[-1]
                next_cursor = encode_cursor(last_product['created_at'], last_product['id'])
            
            response = {
                'products': product_list,
                'limit': limit,
                'next_cursor': next_cursor
            }
            
        else:
            # Offset-based pagination (backward compatible)
            offset = (page - 1) * limit
            
            # Get total count
            cursor.execute('SELECT COUNT(*) as total FROM products')
            total = cursor.fetchone()['total']
            
            # Get products with offset
            query = 'SELECT * FROM products ORDER BY created_at, id LIMIT ? OFFSET ?'
            cursor.execute(query, (limit, offset))
            products = cursor.fetchall()
            
            product_list = []
            for product in products:
                product_dict = {
                    'id': product['id'],
                    'name': product['name'],
                    'price': product['price'],
                    'created_at': product['created_at'],
                    'category': product['category']
                }
                product_list.append(product_dict)
            
            response = {
                'products': product_list,
                'page': page,
                'limit': limit,
                'total': total
            }
        
        conn.close()
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)