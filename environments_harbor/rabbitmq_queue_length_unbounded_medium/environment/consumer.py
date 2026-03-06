#!/usr/bin/env python3

import pika
import yaml
import json
import logging
import time
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Consumer:
    """RabbitMQ Consumer for order processing"""
    
    def __init__(self, config_file='consumer_config.yaml'):
        """Initialize consumer with configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
        
        # Setup connection parameters
        self.connection = None
        self.channel = None
        self.setup_connection_parameters()
    
    def setup_connection_parameters(self):
        """Setup RabbitMQ connection parameters from config"""
        rabbitmq_config = self.config.get('rabbitmq', {})
        self.credentials = pika.PlainCredentials(
            rabbitmq_config.get('username', 'guest'),
            rabbitmq_config.get('password', 'guest')
        )
        self.parameters = pika.ConnectionParameters(
            host=rabbitmq_config.get('host', 'localhost'),
            port=rabbitmq_config.get('port', 5672),
            virtual_host=rabbitmq_config.get('vhost', '/'),
            credentials=self.credentials
        )
    
    def connect(self):
        """Establish connection to RabbitMQ server"""
        try:
            logger.info("Connecting to RabbitMQ...")
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            logger.info("Successfully connected to RabbitMQ")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def on_message(self, channel, method, properties, body):
        """Callback method to process incoming messages"""
        try:
            # Log message receipt
            logger.info(f"Received message: {body.decode()[:100]}...")
            
            # Parse message body
            message_data = json.loads(body.decode())
            logger.info(f"Processing order: {message_data.get('order_id', 'unknown')}")
            
            # Simulate message processing
            time.sleep(0.1)
            
            # Acknowledge message after successful processing
            channel.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Message processed and acknowledged: {method.delivery_tag}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
            # Acknowledge anyway to remove malformed message
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # In production, might want to nack and requeue or send to DLQ
            channel.basic_ack(delivery_tag=method.delivery_tag)
    
    def start_consuming(self):
        """Start consuming messages from the queue"""
        try:
            # Get queue configuration
            queue_config = self.config.get('queue', {})
            queue_name = queue_config.get('name', 'default_queue')
            
            # Declare queue (idempotent operation)
            self.channel.queue_declare(queue=queue_name, durable=True)
            logger.info(f"Queue declared: {queue_name}")
            
            # Set QoS - prefetch count
            prefetch_count = self.config.get('consumer', {}).get('prefetch_count', 1)
            self.channel.basic_qos(prefetch_count=prefetch_count)
            logger.info(f"QoS set with prefetch_count: {prefetch_count}")
            
            # Start consuming messages
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=self.on_message,
                auto_ack=False
            )
            
            logger.info(f"Started consuming from queue: {queue_name}")
            logger.info("Waiting for messages. To exit press CTRL+C")
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Error in start_consuming: {e}")
            self.stop()
    
    def stop(self):
        """Stop consumer and close connections"""
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
                self.channel.close()
            if self.connection and self.connection.is_open:
                self.connection.close()
            logger.info("Consumer stopped and connections closed")
        except Exception as e:
            logger.error(f"Error stopping consumer: {e}")


def main():
    """Main entry point"""
    consumer = Consumer('consumer_config.yaml')
    
    if consumer.connect():
        consumer.start_consuming()
    else:
        logger.error("Failed to start consumer due to connection error")
        sys.exit(1)


if __name__ == '__main__':
    main()