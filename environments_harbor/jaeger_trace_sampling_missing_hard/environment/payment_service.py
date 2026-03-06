#!/usr/bin/env python3

import json
import logging
import random
from flask import Flask, request, jsonify
from jaeger_client import Config
from opentracing.ext import tags
from opentracing import Format
import opentracing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service name - critical for tracing
SERVICE_NAME = 'payment-processing-service'

# Jaeger already integrated - load configuration
def initialize_tracer():
    """Initialize Jaeger tracer with configuration from file"""
    try:
        with open('/app/jaeger_config.json', 'r') as f:
            config_data = json.load(f)
        
        config = Config(
            config=config_data,
            service_name=SERVICE_NAME,
            validate=True
        )
        
        tracer = config.initialize_tracer()
        opentracing.set_global_tracer(tracer)
        logger.info(f"Jaeger tracer initialized for service: {SERVICE_NAME}")
        return tracer
    except Exception as e:
        logger.error(f"Failed to initialize Jaeger tracer: {e}")
        return None

# Initialize tracer
tracer = initialize_tracer()

# Create Flask application
app = Flask(__name__)

# Error types for simulation
ERROR_TYPES = [
    'network timeout',
    'validation error',
    'external service failure'
]

@app.before_request
def start_span():
    """Create a span for each incoming request"""
    if tracer:
        try:
            span_ctx = tracer.extract(Format.HTTP_HEADERS, request.headers)
            span = tracer.start_span(
                operation_name=f"{request.method} {request.path}",
                child_of=span_ctx
            )
            
            # Tag the span with HTTP metadata
            span.set_tag(tags.HTTP_METHOD, request.method)
            span.set_tag(tags.HTTP_URL, request.url)
            span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_SERVER)
            
            # Store span in request context
            request.span = span
        except Exception as e:
            logger.error(f"Error starting span: {e}")
            request.span = None

@app.after_request
def finish_span(response):
    """Finish the span and tag with response status"""
    if tracer and hasattr(request, 'span') and request.span:
        try:
            span = request.span
            span.set_tag(tags.HTTP_STATUS_CODE, response.status_code)
            
            # Tag errors
            if response.status_code >= 400:
                span.set_tag(tags.ERROR, True)
                span.log_kv({'event': 'error', 'status_code': response.status_code})
            
            span.finish()
        except Exception as e:
            logger.error(f"Error finishing span: {e}")
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'service': SERVICE_NAME,
        'tracing_enabled': tracer is not None
    }), 200

@app.route('/api/payment', methods=['POST'])
def process_payment():
    """Process payment request with ~0.5% random failure rate"""
    span = getattr(request, 'span', None)
    
    try:
        # Get payment data
        payment_data = request.get_json()
        
        if not payment_data:
            raise ValueError("No payment data provided")
        
        # Simulate processing
        payment_id = payment_data.get('payment_id', 'unknown')
        amount = payment_data.get('amount', 0)
        
        logger.info(f"Processing payment {payment_id} for amount {amount}")
        
        if span:
            span.set_tag('payment.id', payment_id)
            span.set_tag('payment.amount', amount)
        
        # Simulate 0.5% failure rate
        if random.random() < 0.005:
            error_type = random.choice(ERROR_TYPES)
            logger.error(f"Payment {payment_id} failed: {error_type}")
            
            if span:
                span.set_tag(tags.ERROR, True)
                span.log_kv({
                    'event': 'error',
                    'error.kind': error_type,
                    'message': f'Payment processing failed due to {error_type}',
                    'payment.id': payment_id
                })
            
            return jsonify({
                'status': 'failed',
                'error': error_type,
                'payment_id': payment_id
            }), 500
        
        # Success case
        logger.info(f"Payment {payment_id} processed successfully")
        
        if span:
            span.log_kv({
                'event': 'payment_processed',
                'payment.id': payment_id,
                'status': 'success'
            })
        
        return jsonify({
            'status': 'success',
            'payment_id': payment_id,
            'amount': amount,
            'transaction_id': f'txn_{random.randint(100000, 999999)}'
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        
        if span:
            span.set_tag(tags.ERROR, True)
            span.log_kv({
                'event': 'error',
                'error.kind': 'validation error',
                'message': str(e)
            })
        
        return jsonify({
            'status': 'failed',
            'error': 'validation error',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error processing payment: {e}")
        
        if span:
            span.set_tag(tags.ERROR, True)
            span.log_kv({
                'event': 'error',
                'error.kind': 'internal error',
                'message': str(e)
            })
        
        return jsonify({
            'status': 'failed',
            'error': 'internal error'
        }), 500

if __name__ == '__main__':
    logger.info(f"Starting {SERVICE_NAME} on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)