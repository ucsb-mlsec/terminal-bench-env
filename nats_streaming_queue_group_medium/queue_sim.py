#!/usr/bin/env python3

import os
import sys
import fcntl
import time

def claim_message(message_id, worker_id, claimed_file='/tmp/claimed_messages.txt'):
    """
    Atomically claim a message for this worker.
    Returns True if claim was successful, False if already claimed.
    """
    # Use file locking to ensure atomic operations
    lock_file = '/tmp/queue_claim.lock'
    
    with open(lock_file, 'w') as lock:
        # Acquire exclusive lock
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        
        try:
            # Read already claimed messages
            claimed_messages = set()
            if os.path.exists(claimed_file):
                with open(claimed_file, 'r') as f:
                    claimed_messages = set(line.strip() for line in f if line.strip())
            
            # Check if message is already claimed
            if message_id in claimed_messages:
                return False
            
            # Claim the message by adding it to the claimed file
            with open(claimed_file, 'a') as f:
                f.write(f"{message_id}\n")
            
            return True
            
        finally:
            # Release lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def process_messages(worker_id):
    """
    Process messages from the queue for this worker.
    Each message is claimed atomically to prevent duplicate processing.
    """
    queue_file = '/tmp/message_queue.txt'
    results_file = '/tmp/worker_results.txt'
    results_lock = '/tmp/results.lock'
    
    # Check if queue file exists
    if not os.path.exists(queue_file):
        print(f"[{worker_id}] Queue file not found: {queue_file}")
        return
    
    # Read all messages from queue
    with open(queue_file, 'r') as f:
        messages = [line.strip() for line in f if line.strip()]
    
    print(f"[{worker_id}] Found {len(messages)} messages in queue")
    
    processed_count = 0
    
    # Try to claim and process each message
    for message_id in messages:
        # Attempt to claim the message
        if claim_message(message_id, worker_id):
            print(f"[{worker_id}] Processing {message_id}")
            
            # Simulate some processing time
            time.sleep(0.01)
            
            # Write result to results file (with locking)
            with open(results_lock, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    with open(results_file, 'a') as f:
                        f.write(f"{message_id},{worker_id}\n")
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            
            processed_count += 1
        else:
            # Message already claimed by another worker
            pass
    
    print(f"[{worker_id}] Processed {processed_count} messages")

def main():
    if len(sys.argv) != 2:
        print("Usage: python worker.py <worker_id>")
        sys.exit(1)
    
    worker_id = sys.argv[1]
    
    print(f"[{worker_id}] Starting worker")
    process_messages(worker_id)
    print(f"[{worker_id}] Worker finished")

if __name__ == "__main__":
    main()