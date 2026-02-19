#!/usr/bin/env python3

import numpy as np
import time
import sys

def main():
    print("Memory Access Benchmark - Baseline")
    print("=" * 50)
    
    # Allocate 1GB array (1024 MB)
    array_size = (1024 * 1024 * 1024) // 8  # float64 is 8 bytes
    array_mb = (array_size * 8) // (1024 * 1024)
    
    print(f"Allocating array: {array_mb} MB")
    start_alloc = time.time()
    data = np.zeros(array_size, dtype=np.float64)
    alloc_time = time.time() - start_alloc
    print(f"Allocation time: {alloc_time:.2f} seconds")
    print()
    
    # Sequential access test
    print("Running sequential access test...")
    start_seq = time.time()
    for i in range(len(data)):
        data[i] = data[i] * 1.5 + 3.14
    seq_time = time.time() - start_seq
    print(f"Sequential access: {seq_time:.2f} seconds")
    print()
    
    # Random access test
    print("Running random access test...")
    np.random.seed(42)  # Fixed seed for reproducibility
    num_accesses = 10_000_000
    random_indices = np.random.randint(0, array_size, size=num_accesses)
    
    start_rand = time.time()
    for idx in random_indices:
        data[idx] = data[idx] * 1.5 + 3.14
    rand_time = time.time() - start_rand
    print(f"Random access: {rand_time:.2f} seconds")
    print()
    
    # Summary
    total_time = seq_time + rand_time
    print("=" * 50)
    print("Summary:")
    print(f"  Array size: {array_mb} MB")
    print(f"  Sequential access time: {seq_time:.2f} seconds")
    print(f"  Random access time: {rand_time:.2f} seconds")
    print(f"  Total execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()