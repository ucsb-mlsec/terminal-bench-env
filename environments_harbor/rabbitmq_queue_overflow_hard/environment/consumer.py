#!/usr/bin/env python3

import pika
import json
import time
import os
import sys

def load_config():
    """Load broker configuration from JSON file"""
    config_path = '/home/user/message_processor/broker_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        sys.exit(1)

def process_message(ch, method, properties, body):
    """
    Callback function to process incoming messages.
    Simulates work with 100ms delay per message.
    """
    try:
        message_data = json.loads(body)
        message_id = message_data.get('id', 'unknown')
        
        # Simulate processing work (100ms per message)
        time.sleep(0.1)
        
        print(f'Processed message: {message_id}')
        
        # Acknowledge message after successful processing
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print(f'Error processing message: {e}')
        # Requeue message on error
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def main():
    # Load broker configuration
    config = load_config()
    
    # Extract connection parameters
    host = config.get('host', 'localhost')
    port = config.get('port', 5672)
    username = config.get('username', 'guest')
    password = config.get('password', 'guest')
    
    # Create connection credentials
    credentials = pika.PlainCredentials(username, password)
    parameters = pika.ConnectionParameters(
        host=host,
        port=port,
        credentials=credentials
    )
    
    # Establish connection to RabbitMQ
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    
    # Declare the queue (idempotent operation)
    channel.queue_declare(queue='task_queue', durable=True)
    
    # CRITICAL CONFIGURATION: Prefetch count controls how many unacknowledged messages
    # the consumer will fetch from the broker at once.
    # WARNING: This value is set dangerously high! Will cause memory exhaustion!
    prefetch_count = 1000  # PROBLEMATIC VALUE - fetches too many messages at once
    channel.basic_qos(prefetch_count=prefetch_count)
    
    print(f'Consumer started with prefetch_count={prefetch_count}')
    print('Waiting for messages. To exit press CTRL+C')
    
    # Set up consumer with manual acknowledgment
    channel.basic_consume(
        queue='task_queue',
        on_message_callback=process_message,
        auto_ack=False  # Manual acknowledgment for reliability
    )
    
    # Start consuming messages
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print('\nShutting down consumer gracefully...')
        channel.stop_consuming()
    finally:
        connection.close()
        print('Connection closed')

if __name__ == '__main__':
    main()