#!/usr/bin/env python3
import pika
import json
from datetime import datetime

def callback(ch, method, properties, body):
    """Handle received messages"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    alert_data = json.loads(body)
    
    log_entry = f"[{timestamp}] Station 1 received alert: {json.dumps(alert_data)}\n"
    
    # Write to log file
    with open('station_1.log', 'a') as f:
        f.write(log_entry)
    
    # Print to console
    print(f"[{timestamp}] Station 1 received alert: {alert_data}")

def main():
    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        
        # Declare exchange (matching broker's configuration)
        channel.exchange_declare(exchange='alerts', exchange_type='direct')
        
        # Create exclusive queue
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        
        # Bind queue to exchange
        channel.queue_bind(exchange='alerts', queue=queue_name)
        
        print('Station 1 is waiting for alerts. To exit press CTRL+C')
        
        # Set up consumer
        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
        
        # Start consuming
        channel.start_consuming()
        
    except KeyboardInterrupt:
        print('\nStation 1 shutting down...')
        try:
            channel.stop_consuming()
            connection.close()
        except:
            pass
    except Exception as e:
        print(f'Error in Station 1: {e}')

if __name__ == '__main__':
    main()