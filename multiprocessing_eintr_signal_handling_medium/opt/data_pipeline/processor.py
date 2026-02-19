#!/usr/bin/env python3

import multiprocessing
import queue
import time
import sys
import os
import signal

def process_worker(work_queue, result_queue, worker_id):
    """Worker process that consumes items from the queue and processes them."""
    processed = 0
    while True:
        try:
            # Vulnerable to EINTR - no retry logic
            item = work_queue.get(timeout=2)
            
            if item is None:  # Poison pill
                break
            
            # Simulate some work
            time.sleep(0.15)
            result = item * 2
            
            # Vulnerable to EINTR - no retry logic
            result_queue.put({
                'worker_id': worker_id,
                'item': item,
                'result': result
            })
            
            processed += 1
            
        except queue.Empty:
            continue
    
    # Send final count - vulnerable to EINTR
    result_queue.put({'worker_id': worker_id, 'processed': processed, 'done': True})
    return processed

def result_collector(result_queue, total_items, output_file):
    """Collects results from workers and writes final count."""
    collected = 0
    workers_done = 0
    total_workers = 4
    
    while workers_done < total_workers:
        # Vulnerable to EINTR - no retry logic
        result = result_queue.get()
        
        if 'done' in result and result['done']:
            workers_done += 1
        else:
            collected += 1
    
    # Vulnerable to EINTR during file operations
    with open(output_file, 'w') as f:
        f.write(str(collected))
    
    return collected

def coordinator(items, num_workers=4):
    """Main coordinator that distributes work to workers."""
    work_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    
    # Start worker processes
    workers = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=process_worker,
            args=(work_queue, result_queue, i)
        )
        p.start()
        workers.append(p)
    
    # Start result collector
    collector = multiprocessing.Process(
        target=result_collector,
        args=(result_queue, len(items), '/tmp/processed_count.txt')
    )
    collector.start()
    
    # Distribute work - vulnerable to EINTR
    for item in items:
        work_queue.put(item)
    
    # Send poison pills
    for _ in range(num_workers):
        work_queue.put(None)
    
    # Wait for workers to complete - vulnerable to EINTR
    for worker in workers:
        worker.join()
    
    # Wait for collector - vulnerable to EINTR
    collector.join()
    
    return True

def setup_signal_handlers():
    """Setup signal handlers (currently doesn't help with EINTR)."""
    def signal_handler(signum, frame):
        print(f"Received signal {signum}")
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def main():
    """Main entry point."""
    # Get number of items from command line or use default
    num_items = 100
    if len(sys.argv) > 1:
        try:
            num_items = int(sys.argv[1])
        except ValueError:
            print("Invalid argument, using default 100 items")
    
    setup_signal_handlers()
    
    print(f"Starting data processing pipeline with {num_items} items...")
    
    # Create work items
    items = list(range(1, num_items + 1))
    
    # Run coordinator
    success = coordinator(items, num_workers=4)
    
    # Verify results - vulnerable to EINTR
    if os.path.exists('/tmp/processed_count.txt'):
        with open('/tmp/processed_count.txt', 'r') as f:
            count = int(f.read().strip())
        print(f"Successfully processed {count} items")
        
        if count == num_items:
            print("All items processed successfully!")
            return 0
        else:
            print(f"ERROR: Expected {num_items} but processed {count}")
            return 1
    else:
        print("ERROR: Output file not created")
        return 1

if __name__ == '__main__':
    sys.exit(main())