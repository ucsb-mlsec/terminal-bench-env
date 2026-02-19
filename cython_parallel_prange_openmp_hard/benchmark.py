#!/usr/bin/env python3

import time
import sys

try:
    import numlib
except ImportError as e:
    print(f"Error: Could not import numlib: {e}")
    print("Make sure the module is built and installed correctly.")
    sys.exit(1)

def main():
    print("=" * 60)
    print("NumLib Parallel Processing Benchmark")
    print("=" * 60)
    
    # Check if parallel processing is enabled
    parallel_enabled = numlib.is_parallel_enabled()
    
    if parallel_enabled:
        print("\n✓ Parallel processing: ENABLED")
        print("  Multi-core execution is active")
    else:
        print("\n✗ Parallel processing: DISABLED")
        print("  Code is running sequentially")
    
    print("\n" + "-" * 60)
    print("Running benchmark...")
    print("-" * 60)
    
    # Create test data
    data = list(range(100))
    num_iterations = 5
    
    # Benchmark the computation
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        result = numlib.compute_parallel(data)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Iteration {i+1}/{num_iterations}: {elapsed:.6f} seconds")
    
    # Calculate average time
    avg_time = sum(times) / len(times)
    print(f"\nAverage execution time: {avg_time:.6f} seconds")
    
    print("\n" + "=" * 60)
    if parallel_enabled:
        print("✓ SUCCESS: Multi-core parallel execution confirmed!")
        print("  The module is utilizing multiple CPU cores.")
    else:
        print("✗ WARNING: Running in sequential mode!")
        print("  The module is NOT utilizing multiple CPU cores.")
        print("  Check build configuration and compiler flags.")
    print("=" * 60)
    
    # Exit with appropriate status code
    sys.exit(0 if parallel_enabled else 1)

if __name__ == "__main__":
    main()