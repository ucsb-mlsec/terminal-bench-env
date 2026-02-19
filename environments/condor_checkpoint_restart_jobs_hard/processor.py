#!/usr/bin/env python3

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='Process iterative numerical computations')
    parser.add_argument('--iterations', type=int, required=True, help='Total number of iterations to process')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file (not used in this version)')
    
    args = parser.parse_args()
    
    total_iterations = args.iterations
    result = 0.0
    
    print(f"Starting computation with {total_iterations} iterations")
    start_time = time.time()
    
    for i in range(total_iterations):
        # Perform numerical computation
        result += (i ** 2) / 1000.0
        
        # Print progress periodically
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: iteration {i + 1}/{total_iterations} (elapsed: {elapsed:.2f}s)")
        
        # Small delay to simulate computation time (optional)
        # Uncomment the next line if you want to slow down execution for testing
        # time.sleep(0.0001)
    
    elapsed = time.time() - start_time
    print(f"Computation complete in {elapsed:.2f}s")
    print(f"RESULT: {result}")


if __name__ == "__main__":
    main()