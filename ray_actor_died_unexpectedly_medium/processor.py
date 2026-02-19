#!/usr/bin/env python3

import json
import os
import multiprocessing
from multiprocessing import Pool

def process_file(filepath):
    """Process a single JSON file and return the sum of numbers."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Sum all numbers in the list
    total = sum(data)
    return total

def main():
    data_dir = '/workspace/data/'
    files = [os.path.join(data_dir, f'data_{i}.json') for i in range(10)]
    
    # Create a pool of 4 worker processes
    pool = Pool(processes=4)
    
    # Submit all files for processing
    results = []
    for filepath in files:
        result = pool.apply_async(process_file, (filepath,))
        results.append((filepath, result))
    
    pool.close()
    
    # Collect results - THIS IS WHERE IT HANGS
    # If a worker crashes, get() will wait forever
    processed = 0
    failed = 0
    total_sum = 0
    
    for filepath, result in results:
        try:
            file_sum = result.get()  # No timeout - waits forever if worker crashed
            processed += 1
            total_sum += file_sum
            print(f"Processed {filepath}: {file_sum}")
        except Exception as e:
            failed += 1
            print(f"Failed to process {filepath}: {e}")
    
    pool.join()
    
    # Write results
    with open('/workspace/result.txt', 'w') as f:
        f.write(f"processed={processed}\n")
        f.write(f"failed={failed}\n")
        f.write(f"total_sum={total_sum}\n")
    
    print(f"\nSummary: {processed} processed, {failed} failed, total sum: {total_sum}")

if __name__ == '__main__':
    main()