#!/usr/bin/env python3

import pika
import json
import sys
from datetime import datetime

def send_alert():
    try:
        # Connect to RabbitMQ on localhost
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        
        # Create test alert message
        alert = {
            'alert_id': 'TEST-001',
            'message': 'System health check alert',
            'severity': 'INFO',
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert alert to JSON string
        message_json = json.dumps(alert)
        
        # Publish the alert to the 'alerts' exchange with empty routing_key
        channel.basic_publish(
            exchange='alerts',
            routing_key='',
            body=message_json
        )
        
        print(f"Alert sent: {alert['alert_id']} - {alert['message']}")
        print(f"Severity: {alert['severity']}, Timestamp: {alert['timestamp']}")
        
        # Close the connection
        connection.close()
        
    except pika.exceptions.AMQPConnectionError as e:
        print(f"Error: Could not connect to RabbitMQ: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error sending alert: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    send_alert()