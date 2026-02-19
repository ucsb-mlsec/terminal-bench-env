#!/usr/bin/env python3

import pika
import sys
import time

def start_broker():
    """
    Start the message broker for alert distribution.
    Sets up a fanout exchange to broadcast alerts to all monitoring stations.
    """
    try:
        # Connect to RabbitMQ on localhost
        print("Connecting to RabbitMQ...")
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', port=5672)
        )
        channel = connection.channel()
        
        # Declare a fanout exchange for broadcasting alerts to all stations
        # NOTE: This was previously set to 'direct' which caused messages
        # to only reach one station. Changed to 'fanout' for broadcast behavior.
        channel.exchange_declare(
            exchange='alerts',
            exchange_type='fanout',
            durable=True
        )
        
        print("✓ Alert broker started successfully")
        print("✓ Exchange 'alerts' declared with type 'fanout'")
        print("✓ Broker is ready to distribute alerts to all monitoring stations")
        print("\nPress CTRL+C to stop the broker")
        
        # Keep the broker running
        while True:
            time.sleep(1)
            
    except pika.exceptions.AMQPConnectionError as e:
        print(f"✗ Failed to connect to RabbitMQ: {e}", file=sys.stderr)
        print("Make sure RabbitMQ server is running on localhost:5672", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nShutting down broker...")
        if 'connection' in locals() and connection.is_open:
            connection.close()
        print("✓ Broker stopped")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    start_broker()