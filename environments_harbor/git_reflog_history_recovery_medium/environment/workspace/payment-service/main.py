#!/usr/bin/env python3
"""
Main service file for the Payment Service application.
This is the core Flask application handling payment processing endpoints.
"""

import logging
from datetime import datetime
from flask import Flask, jsonify, request

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify service availability.
    """
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'service': 'payment-service',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/api/payments', methods=['GET'])
def get_payments():
    """
    Retrieve payment records endpoint.
    """
    logger.info("Fetching payment records")
    return jsonify({
        'payments': [],
        'count': 0,
        'message': 'No payments found'
    }), 200


@app.route('/api/payments', methods=['POST'])
def create_payment():
    """
    Create a new payment transaction.
    """
    logger.info("Payment creation requested")
    payment_data = request.get_json()
    
    return jsonify({
        'status': 'success',
        'message': 'Payment processed',
        'data': payment_data
    }), 201


if __name__ == '__main__':
    logger.info("Starting Payment Service...")
    app.run(host='0.0.0.0', port=5000, debug=True)