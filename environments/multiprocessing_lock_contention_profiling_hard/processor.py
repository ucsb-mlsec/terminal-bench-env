#!/usr/bin/env python3

import multiprocessing
import time
import argparse
from multiprocessing import Pool, Semaphore

# CRITICAL BOTTLENECK: This semaphore with value 1 serializes all work
process_semaphore = None

def init_worker(sem):
    """Initialize worker with shared semaphore"""
    global process_semaphore
    process_semaphore = sem

def process_item(item):
    """
    Process a single data item.
    The semaphore prevents parallel execution even with multiple workers.
    """
    # BOTTLENECK: Acquiring semaphore before doing work serializes everything
    process_semaphore.acquire()
    try:
        # Simulate CPU work
        result = 0
        for i in range(10000):
            result += item * i
        
        # Simulate some I/O or processing delay
        time.sleep(0.01)
        
        return result
    finally:
        process_semaphore.release()

def main():
    parser = argparse.ArgumentParser(description='Parallel data processor')
    parser.add_argument('--workers', type=int, default=2, help='Number of worker processes')
    args = parser.parse_args()
    
    # Create 1000 data records to process
    data_records = list(range(1000))
    
    print(f"Processing {len(data_records)} records with {args.workers} workers...")
    
    # Create a semaphore with value 1 - this is the bottleneck!
    # Only one worker can process at a time despite having multiple workers
    sem = Semaphore(1)
    
    start_time = time.time()
    
    # Create pool and process items
    with Pool(processes=args.workers, initializer=init_worker, initargs=(sem,)) as pool:
        results = pool.map(process_item, data_records)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Processing completed in {elapsed:.2f} seconds")
    print(f"Processed {len(results)} items")

if __name__ == '__main__':
    main()