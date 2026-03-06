#!/usr/bin/env python3

from flask import Flask, jsonify
from app.middleware.compression import CompressionMiddleware
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# Register compression middleware
app.wsgi_app = CompressionMiddleware(app.wsgi_app)

def generate_record(record_id):
    """Generate a single analytics record"""
    event_types = ['click', 'view', 'purchase', 'signup', 'logout']
    base_time = datetime.now() - timedelta(days=30)
    
    return {
        'id': record_id,
        'timestamp': (base_time + timedelta(hours=record_id)).isoformat(),
        'user_id': f'user_{random.randint(1000, 9999)}',
        'event_type': random.choice(event_types),
        'value': round(random.uniform(10.0, 1000.0), 2),
        'metadata': {
            'source': random.choice(['web', 'mobile', 'api']),
            'browser': random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),
            'country': random.choice(['US', 'UK', 'CA', 'AU', 'DE']),
            'session_id': f'sess_{random.randint(10000, 99999)}'
        }
    }

@app.route('/api/data/small')
def get_small_data():
    """Returns small JSON response (~500 bytes)"""
    data = [generate_record(i) for i in range(5)]
    return jsonify({
        'status': 'success',
        'size': 'small',
        'data': data
    })

@app.route('/api/data/medium')
def get_medium_data():
    """Returns medium JSON response (~5KB)"""
    data = [generate_record(i) for i in range(50)]
    return jsonify({
        'status': 'success',
        'size': 'medium',
        'data': data
    })

@app.route('/api/data/large')
def get_large_data():
    """Returns large JSON response (~50KB)"""
    data = [generate_record(i) for i in range(500)]
    return jsonify({
        'status': 'success',
        'size': 'large',
        'data': data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)