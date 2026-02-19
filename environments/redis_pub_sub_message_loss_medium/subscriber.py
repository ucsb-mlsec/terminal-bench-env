#!/usr/bin/env python3

import redis
import time
import sys

def main():
    try:
        # Create Redis client connection
        print("Connecting to Redis server at localhost:6379...")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test connection
        r.ping()
        print("Successfully connected to Redis!")
        
        # Simulate connection setup time
        time.sleep(0.5)
        
        # Create pubsub object and subscribe
        pubsub = r.pubsub()
        pubsub.subscribe('system_logs')
        
        print("Subscribed to 'system_logs' channel. Waiting for messages...")
        print("Press Ctrl+C to stop.")
        
        message_count = 0
        
        # Listen for messages
        for message in pubsub.listen():
            if message['type'] == 'message':
                message_count += 1
                print(f"Message {message_count} received: {message['data']}")
            elif message['type'] == 'subscribe':
                print(f"Subscription confirmed to channel: {message['channel']}")
                
    except redis.ConnectionError as e:
        print(f"ERROR: Could not connect to Redis server: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nShutting down subscriber. Total messages received: {message_count}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()