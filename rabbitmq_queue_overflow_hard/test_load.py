#!/usr/bin/env python3

import pika
import json
import sys
import time

def load_broker_config():
    """Load broker configuration from JSON file."""
    with open('/home/user/message_processor/broker_config.json', 'r') as f:
        return json.load(f)

def create_connection(config):
    """Create connection to RabbitMQ using broker config."""
    credentials = pika.PlainCredentials(
        config.get('username', 'guest'),
        config.get('password', 'guest')
    )
    parameters = pika.ConnectionParameters(
        host=config.get('host', 'localhost'),
        port=config.get('port', 5672),
        credentials=credentials,
        virtual_host=config.get('virtual_host', '/')
    )
    return pika.BlockingConnection(parameters)

def publish_messages(channel, num_messages=1000):
    """
    Publish specified number of messages to the task_queue.
    
    Args:
        channel: Pika channel object
        num_messages: Number of messages to publish (default 1000)
    """
    print(f"Starting to publish {num_messages} messages...")
    start_time = time.time()
    
    for i in range(num_messages):
        message = {
            "message_id": i,
            "data": f"test_payload_{i}",
            "timestamp": time.time()
        }
        
        message_body = json.dumps(message)
        
        # Publish message with persistence enabled
        channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            body=message_body,
            properties=pika.BasicProperties(
                delivery_mode=2  # Make message persistent
            )
        )
        
        # Print progress every 100 messages
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"Published {i + 1}/{num_messages} messages ({rate:.1f} msg/sec)")
    
    elapsed_total = time.time() - start_time
    final_rate = num_messages / elapsed_total
    print(f"\nCompleted publishing {num_messages} messages in {elapsed_total:.2f} seconds")
    print(f"Average rate: {final_rate:.1f} messages/second")

def main():
    """Main function to load config, connect, and publish messages."""
    # Get number of messages from command line argument
    num_messages = 1000
    if len(sys.argv) > 1:
        try:
            num_messages = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of messages: {sys.argv[1]}, using default 1000")
            num_messages = 1000
    
    try:
        # Load broker configuration
        print("Loading broker configuration...")
        config = load_broker_config()
        
        # Create connection
        print("Connecting to RabbitMQ...")
        connection = create_connection(config)
        channel = connection.channel()
        
        # Declare queue with durability
        channel.queue_declare(queue='task_queue', durable=True)
        print("Queue 'task_queue' declared (durable)")
        
        # Publish messages
        publish_messages(channel, num_messages)
        
        # Summary
        print(f"\n{'='*50}")
        print(f"Sent {num_messages} messages to task_queue")
        print(f"{'='*50}")
        
        # Close connection
        connection.close()
        print("Connection closed successfully")
        
    except FileNotFoundError:
        print("Error: Broker configuration file not found at /home/user/message_processor/broker_config.json")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()