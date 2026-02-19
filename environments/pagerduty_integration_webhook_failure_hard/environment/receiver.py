#!/usr/bin/env python3

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
from urllib.parse import urlparse

# Load configuration
def load_config():
    config = {}
    config_path = '/opt/webhook/config.env'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    return config

config = load_config()
PAGERDUTY_ROUTING_KEY = config.get('PAGERDUTY_ROUTING_KEY', '')
PAGERDUTY_URL = config.get('PAGERDUTY_URL', 'http://localhost:8080/v2/enqueue')

class WebhookHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        """Override to log to file instead of stderr"""
        log_file = '/var/log/webhook/requests.log'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write("%s - - [%s] %s\n" %
                    (self.client_address[0],
                     self.log_date_time_string(),
                     format % args))
    
    def do_POST(self):
        """Handle POST requests from monitoring system"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse incoming alert
            alert = json.loads(post_data.decode('utf-8'))
            
            # Log the received alert
            self.log_message("Received alert: %s", json.dumps(alert))
            
            # Forward to PagerDuty
            response_code = self.forward_to_pagerduty(alert)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'forwarded',
                'pagerduty_response': response_code
            }).encode('utf-8'))
            
        except Exception as e:
            self.log_message("Error processing request: %s", str(e))
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
    
    def forward_to_pagerduty(self, alert):
        """Forward alert to PagerDuty Events API v2"""
        # Build PagerDuty payload - BUG: Missing event_action field
        payload = {
            'routing_key': PAGERDUTY_ROUTING_KEY,
            'payload': {
                'summary': f"{alert.get('alert_name', 'Alert')}: {alert.get('message', '')}",
                'severity': alert.get('severity', 'error'),
                'source': alert.get('host', 'unknown'),
                'custom_details': alert
            }
        }
        
        # Send to PagerDuty
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(PAGERDUTY_URL, 
                                    json=payload, 
                                    headers=headers,
                                    timeout=10)
            self.log_message("PagerDuty response: %d", response.status_code)
            return response.status_code
        except Exception as e:
            self.log_message("Failed to forward to PagerDuty: %s", str(e))
            return 500

def run_server(port=8000):
    """Run the webhook server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, WebhookHandler)
    print(f'Webhook receiver running on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()