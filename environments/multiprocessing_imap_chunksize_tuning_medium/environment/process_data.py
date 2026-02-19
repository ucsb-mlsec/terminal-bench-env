#!/usr/bin/env python3

import multiprocessing
import time
import numpy as np

def process_item(x):
    """
    Compute-intensive function that processes a single numerical item.
    Performs multiple mathematical operations to simulate real workload.
    """
    result = x
    for _ in range(10):
        result = np.sqrt(abs(result) + 1)
        result = np.sin(result) * np.cos(result)
        result = result ** 2 + np.log(abs(result) + 1)
    return result

if __name__ == '__main__':
    # Generate test dataset of exactly 100,000 numerical items
    print("Generating test dataset...")
    data = np.arange(100000, dtype=np.float64)
    
    print(f"Processing {len(data)} items with 4 worker processes...")
    
    # Record start time
    start_time = time.time()
    
    # Create multiprocessing Pool with 4 worker processes
    pool = multiprocessing.Pool(processes=4)
    
    # Use pool.imap() WITHOUT specifying chunksize (defaults to 1, causing performance issue)
    results = list(pool.imap(process_item, data))
    
    # Close and join the pool
    pool.close()
    pool.join()
    
    # Record end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print execution time
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Processed {len(results)} items")
    print(f"Sample results: {results[:5]}")