#!/usr/bin/env python3

import pika
import json
from datetime import datetime

def callback(ch, method, properties, body):
    """Handle received alert messages"""
    try:
        alert = json.loads(body)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] Station 3 received alert: {alert}\n"
        
        # Write to log file
        with open('/home/agent/alert_system/station_3.log', 'a') as f:
            f.write(log_entry)
        
        # Print to console
        print(f"[Station 3] Alert received at {timestamp}: {alert}")
        
    except Exception as e:
        print(f"[Station 3] Error processing alert: {e}")

def main():
    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        
        # Declare the exchange (matching broker's configuration)
        channel.exchange_declare(exchange='alerts', exchange_type='direct')
        
        # Create exclusive queue with automatic name
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        
        # Bind queue to exchange
        channel.queue_bind(exchange='alerts', queue=queue_name)
        
        print('[Station 3] Waiting for alerts. To exit press CTRL+C')
        
        # Set up consumer
        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
        
        # Start consuming
        channel.start_consuming()
        
    except KeyboardInterrupt:
        print('\n[Station 3] Shutting down...')
        try:
            connection.close()
        except:
            pass
    except Exception as e:
        print(f'[Station 3] Error: {e}')

if __name__ == '__main__':
    main()