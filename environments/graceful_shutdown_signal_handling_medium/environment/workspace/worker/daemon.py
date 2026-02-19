#!/usr/bin/env python3

import sys
import time
import re
from datetime import datetime

QUEUE_FILE = '/workspace/worker/queue.txt'
RESULTS_FILE = '/workspace/worker/results.txt'

def extract_duration(task_id):
    """Extract sleep duration from task_id like 'task_5' -> 5"""
    match = re.search(r'(\d+)', task_id)
    if match:
        return int(match.group(1))
    return 1  # Default duration

def process_task(task_id):
    """Process a single task by sleeping for the duration encoded in its ID"""
    duration = extract_duration(task_id)
    print(f"Processing {task_id} (will take {duration} seconds)...", flush=True)
    time.sleep(duration)
    
    # Write completion record
    timestamp = datetime.now().isoformat()
    with open(RESULTS_FILE, 'a') as f:
        f.write(f"{task_id},{timestamp}\n")
    
    print(f"Completed {task_id}", flush=True)

def main():
    """Main daemon loop - reads and processes tasks from queue"""
    print("Daemon starting...", flush=True)
    
    # Read all tasks from queue file
    try:
        with open(QUEUE_FILE, 'r') as f:
            tasks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Queue file {QUEUE_FILE} not found", flush=True)
        sys.exit(1)
    
    print(f"Found {len(tasks)} tasks to process", flush=True)
    
    # Process each task sequentially
    for task_id in tasks:
        process_task(task_id)
    
    print("All tasks completed. Daemon exiting.", flush=True)
    sys.exit(0)

if __name__ == '__main__':
    main()