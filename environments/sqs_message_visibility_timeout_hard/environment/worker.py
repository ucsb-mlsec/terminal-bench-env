#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime, timedelta

def log_message(message_id, status, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] MESSAGE_ID: {message_id} | STATUS: {status}"
    
    for key, value in kwargs.items():
        log_entry += f" | {key.upper()}: {value}"
    
    log_entry += "\n"
    
    with open('/home/agent/message_log.txt', 'a') as f:
        f.write(log_entry)

def read_config():
    try:
        with open('/home/agent/config.json', 'r') as f:
            config = json.load(f)
            return config.get('visibility_timeout', 30)
    except Exception as e:
        print(f"Error reading config: {e}")
        return 30

def get_message_files():
    queue_dir = '/home/agent/queue/'
    if not os.path.exists(queue_dir):
        return []
    
    files = []
    for filename in os.listdir(queue_dir):
        if filename.endswith('.json'):
            files.append(os.path.join(queue_dir, filename))
    return files

def process_message(message_file, visibility_timeout):
    try:
        with open(message_file, 'r') as f:
            message = json.load(f)
        
        message_id = message.get('message_id')
        visibility_until = message.get('visibility_until', 0)
        
        current_time = time.time()
        
        if current_time < visibility_until:
            return False
        
        new_visibility = current_time + visibility_timeout
        message['visibility_until'] = new_visibility
        
        with open(message_file, 'w') as f:
            json.dump(message, f)
        
        log_message(message_id, 'PROCESSING', worker_id='worker_1')
        
        start_time = time.time()
        
        processing_time = 45
        time.sleep(processing_time)
        
        end_time = time.time()
        duration = int(end_time - start_time)
        
        log_message(message_id, 'PROCESSED', duration=duration)
        
        os.remove(message_file)
        
        return True
        
    except Exception as e:
        print(f"Error processing message: {e}")
        return False

def main():
    print("Worker started...")
    visibility_timeout = read_config()
    print(f"Visibility timeout: {visibility_timeout} seconds")
    
    while True:
        message_files = get_message_files()
        
        if message_files:
            for message_file in message_files:
                process_message(message_file, visibility_timeout)
        
        time.sleep(1)

if __name__ == '__main__':
    main()