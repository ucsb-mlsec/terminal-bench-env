#!/usr/bin/env python3

import subprocess
import time
import signal
import sys
import os
import random

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle SIGTERM for graceful shutdown"""
    global shutdown_requested
    print(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

def process_message(message_id):
    """
    Spawn a worker process to handle a message.
    Intentionally does NOT wait for the child process - this causes zombies!
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing message: {message_id}")
    
    # Spawn worker process without waiting for it
    # This is the problematic behavior that causes zombie processes
    worker_process = subprocess.Popen(
        ['/bin/bash', 'worker.sh', str(message_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Spawned worker PID {worker_process.pid} for message {message_id}")
    
    # Intentionally NOT calling worker_process.wait() or worker_process.communicate()
    # This leaves the process unreapead when it exits, creating a zombie

def main():
    """Main processing loop"""
    print("Starting message processor service...")
    print(f"Process PID: {os.getpid()}")
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    message_count = 0
    
    # Main processing loop
    while not shutdown_requested:
        message_count += 1
        message_id = f"MSG-{message_count:06d}-{random.randint(1000, 9999)}"
        
        try:
            process_message(message_id)
        except Exception as e:
            print(f"Error processing message {message_id}: {e}")
        
        # Random sleep interval between 1-3 seconds
        sleep_time = random.uniform(1.0, 3.0)
        time.sleep(sleep_time)
    
    print("Shutting down message processor service...")
    print(f"Total messages processed: {message_count}")
    sys.exit(0)

if __name__ == "__main__":
    main()