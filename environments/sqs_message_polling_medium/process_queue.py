#!/usr/bin/env python3

import os
import json
import glob

QUEUE_DIR = '/var/queue/messages/'
OUTPUT_FILE = '/tmp/processed_orders.txt'

def read_message_file(filepath):
    """Read and parse a JSON message file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def process_message(message_data):
    """Extract order information from message"""
    try:
        order_id = message_data.get('order_id')
        customer_name = message_data.get('customer_name')
        amount = message_data.get('amount')
        return order_id, customer_name, amount
    except Exception as e:
        print(f"Error processing message: {e}")
        return None, None, None

def write_order_to_file(order_id, customer_name, amount):
    """Write order data to output file"""
    try:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(f"{order_id},{customer_name},{amount}\n")
        return True
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return False

def process_queue():
    """Process all messages in the queue directory"""
    # Find all JSON message files
    message_files = glob.glob(os.path.join(QUEUE_DIR, '*.json'))
    
    if not message_files:
        print("No messages to process")
        return
    
    print(f"Found {len(message_files)} messages to process")
    
    for message_file in message_files:
        print(f"Processing: {message_file}")
        
        # Read the message
        message_data = read_message_file(message_file)
        
        if message_data:
            # Process the message
            order_id, customer_name, amount = process_message(message_data)
            
            if order_id and customer_name and amount:
                # Write to output file
                write_order_to_file(order_id, customer_name, amount)
                print(f"Processed order {order_id}")
                # BUG: Not deleting the message file after processing!
                # This causes messages to be reprocessed on next run

if __name__ == '__main__':
    process_queue()
