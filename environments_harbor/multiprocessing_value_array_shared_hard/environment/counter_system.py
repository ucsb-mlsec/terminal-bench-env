#!/usr/bin/env python3

import sys
import random
import multiprocessing
import time

def worker(data_chunk, low_counter, medium_counter, high_counter):
    """
    Worker process that counts values in its assigned data chunk.
    This implementation has a critical race condition - it updates shared
    counters without proper synchronization.
    """
    local_low = 0
    local_medium = 0
    local_high = 0
    
    # Count values in local variables first
    for value in data_chunk:
        if 0 <= value <= 33:
            local_low += 1
        elif 34 <= value <= 66:
            local_medium += 1
        elif 67 <= value <= 100:
            local_high += 1
    
    # RACE CONDITION: These updates are not atomic!
    # Multiple processes reading and writing simultaneously cause lost updates
    low_counter.value = low_counter.value + local_low
    medium_counter.value = medium_counter.value + local_medium
    high_counter.value = high_counter.value + local_high

def main():
    # Get number of workers from command line
    num_workers = 4
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    
    # Generate dataset with fixed seed for reproducibility
    dataset_size = 500000
    random.seed(42)
    dataset = [random.randint(0, 100) for _ in range(dataset_size)]
    
    # Create shared counters without locks (this causes race conditions!)
    low_counter = multiprocessing.Value('i', 0)
    medium_counter = multiprocessing.Value('i', 0)
    high_counter = multiprocessing.Value('i', 0)
    
    # Split dataset into chunks for each worker
    chunk_size = dataset_size // num_workers
    chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        if i == num_workers - 1:
            # Last chunk gets any remaining items
            end_idx = dataset_size
        else:
            end_idx = start_idx + chunk_size
        chunks.append(dataset[start_idx:end_idx])
    
    # Start timer
    start_time = time.time()
    
    # Create and start worker processes
    processes = []
    for chunk in chunks:
        p = multiprocessing.Process(
            target=worker,
            args=(chunk, low_counter, medium_counter, high_counter)
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    # Get final counts
    low_count = low_counter.value
    medium_count = medium_counter.value
    high_count = high_counter.value
    total = low_count + medium_count + high_count
    
    # Calculate accuracy
    accuracy = (total / dataset_size) * 100
    
    # Output results
    print(f"LOW={low_count}")
    print(f"MEDIUM={medium_count}")
    print(f"HIGH={high_count}")
    print(f"TOTAL={total}")
    print(f"EXPECTED={dataset_size}")
    print(f"ACCURACY={accuracy:.2f}%")

if __name__ == "__main__":
    main()