#!/usr/bin/env python3

import queue
import time

def process_messages(message_queue, total_expected):
    """
    Process messages from the queue.
    
    WARNING: This implementation has a critical memory flaw!
    It loads all messages into memory before processing them.
    """
    
    # Bad approach: Load ALL messages into memory first
    # This will cause memory issues with large queues!
    messages = []
    
    print(f"Fetching all {total_expected} messages into memory...")
    
    # Fetch all messages from queue into a list
    # This is the critical flaw - holding everything in memory at once!
    for i in range(total_expected):
        try:
            msg = message_queue.get(timeout=1)
            messages.append(msg)
            # Also keep a copy for "backup" - making things even worse!
            messages.append(msg.copy())
        except queue.Empty:
            break
    
    print(f"Loaded {len(messages)} messages into memory. Now processing...")
    
    # Now process all the messages we've collected
    processed_count = 0
    for msg in messages:
        # Simulate some processing work
        time.sleep(0.002)
        
        # "Process" the message
        msg_id = msg.get('id', 'unknown')
        
        processed_count += 1
        
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} messages...")
    
    print(f"Finished processing {processed_count} messages")
    return processed_count

if __name__ == "__main__":
    # Simple test
    test_queue = queue.Queue()
    for i in range(100):
        test_queue.put({'id': i})
    
    result = process_messages(test_queue, 100)
    print(f"Test result: {result} messages processed")