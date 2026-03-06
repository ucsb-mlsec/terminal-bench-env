#!/usr/bin/env python3

from flask import Flask, request, jsonify
import sqlite3
import json
from datetime import datetime
import uuid
import threading

app = Flask(__name__)
DATABASE = 'transactions.db'
db_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE NOT NULL,
            amount REAL NOT NULL,
            currency TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            customer_id TEXT NOT NULL
        )
    ''')
    
    # Create idempotency table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS idempotency_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT UNIQUE NOT NULL,
            request_hash TEXT NOT NULL,
            response_body TEXT NOT NULL,
            response_status INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_idempotency_key 
        ON idempotency_cache(idempotency_key)
    ''')
    
    conn.commit()
    conn.close()

def generate_request_hash(payload):
    """Generate a hash of the request payload for validation"""
    sorted_payload = json.dumps(payload, sort_keys=True)
    return str(hash(sorted_payload))

def get_cached_response(idempotency_key):
    """Retrieve cached response for an idempotency key"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clean up expired keys (older than 24 hours)
    cursor.execute('''
        DELETE FROM idempotency_cache 
        WHERE created_at < datetime('now', '-24 hours')
    ''')
    conn.commit()
    
    # Fetch cached response
    cursor.execute('''
        SELECT response_body, response_status, request_hash
        FROM idempotency_cache
        WHERE idempotency_key = ?
        AND created_at >= datetime('now', '-24 hours')
    ''', (idempotency_key,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'response_body': json.loads(result['response_body']),
            'response_status': result['response_status'],
            'request_hash': result['request_hash']
        }
    return None

def cache_response(idempotency_key, request_hash, response_body, response_status):
    """Cache a response for future idempotent requests"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO idempotency_cache 
            (idempotency_key, request_hash, response_body, response_status)
            VALUES (?, ?, ?, ?)
        ''', (idempotency_key, request_hash, json.dumps(response_body), response_status))
        conn.commit()
    except sqlite3.IntegrityError:
        # Key already exists, which is fine in concurrent scenarios
        pass
    finally:
        conn.close()

@app.route('/api/payment', methods=['POST'])
def create_payment():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['amount', 'currency', 'customer_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get idempotency key from headers
        idempotency_key = request.headers.get('Idempotency-Key')
        
        # Validate idempotency key format if provided
        if idempotency_key:
            if not idempotency_key.strip():
                return jsonify({'error': 'Invalid idempotency key: empty string'}), 400
            if len(idempotency_key) > 255:
                return jsonify({'error': 'Invalid idempotency key: too long (max 255 characters)'}), 400
        
        # Generate request hash for validation
        request_payload = {
            'amount': data['amount'],
            'currency': data['currency'],
            'customer_id': data['customer_id']
        }
        request_hash = generate_request_hash(request_payload)
        
        # Check for cached response if idempotency key is provided
        if idempotency_key:
            with db_lock:
                cached = get_cached_response(idempotency_key)
                
                if cached:
                    # Verify request payload matches
                    if cached['request_hash'] != request_hash:
                        return jsonify({
                            'error': 'Idempotency key reused with different request parameters'
                        }), 422
                    
                    # Return cached response
                    return jsonify(cached['response_body']), cached['response_status']
        
        # Process new transaction
        with db_lock:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            transaction_id = str(uuid.uuid4())
            amount = float(data['amount'])
            currency = data['currency']
            customer_id = data['customer_id']
            status = 'completed'
            timestamp = datetime.utcnow().isoformat()
            
            try:
                cursor.execute('''
                    INSERT INTO transactions 
                    (transaction_id, amount, currency, status, customer_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (transaction_id, amount, currency, status, customer_id, timestamp))
                
                conn.commit()
                
                response_body = {
                    'transaction_id': transaction_id,
                    'amount': amount,
                    'currency': currency,
                    'status': status,
                    'timestamp': timestamp
                }
                response_status = 201
                
                # Cache the response if idempotency key is provided
                if idempotency_key:
                    cache_response(idempotency_key, request_hash, response_body, response_status)
                
                conn.close()
                return jsonify(response_body), response_status
                
            except sqlite3.Error as e:
                conn.close()
                return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/transaction/<transaction_id>', methods=['GET'])
def get_transaction(transaction_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT transaction_id, amount, currency, status, created_at, customer_id
            FROM transactions
            WHERE transaction_id = ?
        ''', (transaction_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            transaction = {
                'transaction_id': row['transaction_id'],
                'amount': row['amount'],
                'currency': row['currency'],
                'status': row['status'],
                'timestamp': row['created_at'],
                'customer_id': row['customer_id']
            }
            return jsonify(transaction), 200
        else:
            return jsonify({'error': 'Transaction not found'}), 404
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)