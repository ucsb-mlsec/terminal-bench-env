#!/usr/bin/env python3

import argparse
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def process_batch(batch_id):
    """
    Simulate data loading and processing for a single batch.
    This includes I/O simulation and computational work.
    """
    # Simulate I/O delay (reading from disk/network)
    time.sleep(random.uniform(0.05, 0.15))
    
    # Simulate computational work (data transformation, normalization)
    data = np.random.randn(1000, 100)
    
    # Normalize
    normalized = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    
    # Apply transformations
    transformed = np.tanh(normalized)
    result = np.mean(transformed ** 2)
    
    # Additional CPU work
    for _ in range(10):
        temp = np.random.randn(100, 100)
        temp = np.dot(temp, temp.T)
    
    return result

def load_data_sync(num_batches):
    """Synchronous data loading (workers=0)"""
    results = []
    for i in range(num_batches):
        result = process_batch(i)
        results.append(result)
    return results

def load_data_parallel(num_batches, num_workers):
    """Parallel data loading using ProcessPoolExecutor"""
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, i) for i in range(num_batches)]
        for future in futures:
            results.append(future.result())
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading performance')
    parser.add_argument('--workers', type=int, default=0,
                      help='Number of parallel worker processes (0 for synchronous)')
    args = parser.parse_args()
    
    num_batches = 80
    num_workers = args.workers
    
    print(f"Starting benchmark with {num_workers} workers...")
    print(f"Processing {num_batches} batches...")
    
    start_time = time.time()
    
    if num_workers == 0:
        results = load_data_sync(num_batches)
    else:
        results = load_data_parallel(num_batches, num_workers)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Processed {len(results)} batches")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per batch: {total_time/num_batches:.3f} seconds")

if __name__ == '__main__':
    main()