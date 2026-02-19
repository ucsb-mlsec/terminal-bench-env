#!/usr/bin/env python3

import json
import logging
from flask import Flask, request, jsonify
import requests
from datetime import datetime

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    filename='/var/log/webhook_service.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = '/opt/webhook_service/config.json'
config = {}

def load_config():
    """Load configuration from config.json"""
    global config
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = {'pagerduty_integration_key': 'default-key'}

def transform_alert_to_pagerduty(alert_data):
    """Transform incoming alert to PagerDuty Events API v2 format"""
    try:
        # Map severity levels
        severity_map = {
            'critical': 'critical',
            'warning': 'warning',
            'info': 'info',
            'error': 'error'
        }
        
        severity = severity_map.get(alert_data.get('severity', 'info'), 'info')
        alert_name = alert_data.get('alert_name', 'Unknown Alert')
        host = alert_data.get('host', 'unknown-host')
        timestamp = alert_data.get('timestamp', datetime.utcnow().isoformat())
        
        # Construct summary
        summary = f"{alert_name} on {host}"
        
        # Build PagerDuty event payload - BUG: using 'route_key' instead of 'routing_key'
        pagerduty_event = {
            'route_key': config.get('pagerduty_integration_key'),
            'event_action': 'trigger',
            'payload': {
                'summary': summary,
                'severity': severity,
                'source': host,
                'timestamp': timestamp,
                'custom_details': {
                    'alert_name': alert_name,
                    'host': host,
                    'original_severity': alert_data.get('severity', 'unknown')
                }
            }
        }
        
        logger.info(f"Transformed alert to PagerDuty format: {json.dumps(pagerduty_event)}")
        return pagerduty_event
        
    except Exception as e:
        logger.error(f"Error transforming alert: {e}")
        return None

def send_to_pagerduty(event_data):
    """Send event to PagerDuty Events API v2"""
    pagerduty_url = 'https://events.pagerduty.com/v2/enqueue'
    
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Sending event to PagerDuty: {pagerduty_url}")
        response = requests.post(
            pagerduty_url,
            json=event_data,
            headers=headers,
            timeout=10
        )
        
        logger.info(f"PagerDuty API response status: {response.status_code}")
        logger.info(f"PagerDuty API response body: {response.text}")
        
        if response.status_code == 202:
            logger.info("Successfully sent event to PagerDuty")
            return True, "Event accepted by PagerDuty"
        else:
            logger.warning(f"PagerDuty API returned non-202 status: {response.status_code}")
            return False, f"PagerDuty API error: {response.text}"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send event to PagerDuty: {e}")
        return False, str(e)

@app.route('/webhook', methods=['POST'])
def receive_webhook():
    """Webhook endpoint to receive alerts"""
    try:
        # Get incoming alert data
        alert_data = request.get_json()
        
        if not alert_data:
            logger.warning("Received empty webhook payload")
            return jsonify({'error': 'Empty payload'}), 400
        
        logger.info(f"Received webhook: {json.dumps(alert_data)}")
        
        # Transform alert to PagerDuty format
        pagerduty_event = transform_alert_to_pagerduty(alert_data)
        
        if not pagerduty_event:
            logger.error("Failed to transform alert")
            return jsonify({'error': 'Failed to transform alert'}), 500
        
        # Send to PagerDuty
        success, message = send_to_pagerduty(pagerduty_event)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Alert forwarded to PagerDuty'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': message
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    load_config()
    logger.info("Starting webhook receiver service")
    app.run(host='0.0.0.0', port=5000, debug=False)