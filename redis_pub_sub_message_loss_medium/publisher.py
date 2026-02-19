#!/usr/bin/env python3

import redis
import json
import time
from datetime import datetime

def main():
    # Create Redis client connection
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("Connected to Redis successfully")
    except redis.ConnectionError as e:
        print(f"Failed to connect to Redis: {e}")
        return
    
    channel = 'system_logs'
    
    # Define log messages with different levels
    log_messages = [
        {"level": "INFO", "message": "Application started successfully"},
        {"level": "INFO", "message": "Database connection pool initialized"},
        {"level": "WARNING", "message": "High memory usage detected: 85%"},
        {"level": "INFO", "message": "User authentication successful"},
        {"level": "ERROR", "message": "Database connection failed"},
        {"level": "WARNING", "message": "API response time exceeds threshold"},
        {"level": "CRITICAL", "message": "Primary database server unreachable"},
        {"level": "ERROR", "message": "Failed to process payment transaction"},
        {"level": "INFO", "message": "Cache cleared successfully"},
        {"level": "WARNING", "message": "Disk space running low: 90% used"},
        {"level": "ERROR", "message": "Network timeout while connecting to service"},
        {"level": "CRITICAL", "message": "System memory critically low"},
        {"level": "INFO", "message": "Backup process completed"},
        {"level": "ERROR", "message": "Invalid configuration file detected"},
        {"level": "INFO", "message": "System health check passed"}
    ]
    
    print(f"\nPublishing {len(log_messages)} log messages to channel '{channel}'...\n")
    
    # Publish messages with delays
    for i, log in enumerate(log_messages, 1):
        # Add timestamp
        log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Convert to JSON string
        message = json.dumps(log)
        
        # Publish message
        subscribers = r.publish(channel, message)
        print(f"[{i:2d}] Published {log['level']:8s} message - {subscribers} subscriber(s) received")
        
        # Add delays between some messages to simulate real-world timing
        if i % 3 == 0:
            time.sleep(0.5)
        elif i % 2 == 0:
            time.sleep(0.3)
        else:
            time.sleep(0.1)
    
    print(f"\nFinished publishing all {len(log_messages)} messages")

if __name__ == "__main__":
    main()