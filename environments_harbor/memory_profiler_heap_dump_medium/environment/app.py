#!/usr/bin/env python3

from flask import Flask, jsonify
from handlers.api import api_blueprint
from handlers.metrics import metrics_blueprint

app = Flask(__name__)

# Register blueprints
app.register_blueprint(api_blueprint, url_prefix='/api')
app.register_blueprint(metrics_blueprint, url_prefix='/metrics')

@app.route('/')
def index():
    """Root endpoint returning welcome message"""
    return jsonify({
        'message': 'Welcome to the Web Application',
        'status': 'running',
        'version': '1.0.0'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)