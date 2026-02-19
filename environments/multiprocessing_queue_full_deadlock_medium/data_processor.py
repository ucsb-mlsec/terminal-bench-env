#!/usr/bin/env python3
"""
Data Processing Pipeline
A parallel data processing system that transforms data records using multiple worker processes.
WARNING: Contains a critical deadlock vulnerability in production use.
"""

import multiprocessing
from multiprocessing import Process, Queue
import json
import sys
import time


def worker_process(input_queue, output_queue):
    """
    Worker process that transforms data records.
    Continuously processes items from input queue and puts results to output queue.
    """
    while True:
        # Get item from input queue
        item = input_queue.get()
        
        # Check for poison pill (termination signal)
        if item is None:
            break
        
        # Process the data record
        # Transform: convert string values to uppercase, multiply numeric values by 2
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, str):
                processed_item[key] = value.upper()
            elif isinstance(value, (int, float)):
                processed_item[key] = value * 2
            else:
                processed_item[key] = value
        
        # Add processing timestamp
        processed_item['processed_at'] = time.time()
        
        # Put result into output queue
        # CRITICAL: This can block if output_queue is full!
        output_queue.put(processed_item)


def process_data(input_file):
    """
    Main data processing function that coordinates the parallel pipeline.
    
    DEADLOCK BUG: This implementation puts all input data into queues before
    collecting results, which can cause deadlock with large datasets when
    the output queue fills up and workers block.
    """
    print(f"Loading data from {input_file}...")
    
    # Load input data
    with open(input_file, 'r') as f:
        data_records = json.load(f)
    
    num_records = len(data_records)
    print(f"Loaded {num_records} records to process")
    
    # Create queues for inter-process communication
    # BUG: Using default maxsize which can lead to blocking
    input_queue = Queue()
    output_queue = Queue()
    
    # Number of worker processes
    num_workers = 4
    
    # Spawn worker processes
    workers = []
    for i in range(num_workers):
        p = Process(target=worker_process, args=(input_queue, output_queue))
        p.start()
        workers.append(p)
        print(f"Started worker process {i+1}")
    
    print("Feeding data to workers...")
    
    # CRITICAL SECTION: Put ALL input items into queue BEFORE collecting results
    # This is the deadlock-prone pattern!
    for record in data_records:
        input_queue.put(record)
    
    print("All input data queued")
    
    # Send poison pills to workers
    for _ in range(num_workers):
        input_queue.put(None)
    
    print("Poison pills sent, collecting results...")
    
    # Now try to collect all results
    # DEADLOCK: If output_queue filled up during processing, workers are blocked
    # on put() and we're blocked here trying to get(), but workers can't proceed
    results = []
    for _ in range(num_records):
        result = output_queue.get()
        results.append(result)
    
    print("All results collected, waiting for workers to finish...")
    
    # Wait for all workers to complete
    for p in workers:
        p.join()
    
    print(f"Processing complete: {len(results)} records processed")
    
    return results


if __name__ == '__main__':
    # Get input file from command line or use default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '/home/user/pipeline/test_data.json'
    
    start_time = time.time()
    
    try:
        results = process_data(input_file)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"Pipeline execution completed successfully")
        print(f"Records processed: {len(results)}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)