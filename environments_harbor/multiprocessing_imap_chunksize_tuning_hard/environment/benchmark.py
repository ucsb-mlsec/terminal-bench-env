#!/usr/bin/env python3

import subprocess
import time
import json
import sys
import os

def run_benchmark(filepath, chunk_size=None):
    """
    Run process_transactions.py with given parameters and measure execution time.
    
    Args:
        filepath: Path to the CSV file to process
        chunk_size: Optional chunk size parameter
    
    Returns:
        Execution time in seconds
    """
    command = [sys.executable, 'process_transactions.py', filepath]
    
    if chunk_size is not None:
        command.append(str(chunk_size))
    
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if result.returncode != 0:
        print(f"Warning: Process returned non-zero exit code: {result.returncode}")
        if result.stderr:
            print(f"Error output: {result.stderr}")
    
    return execution_time

def test_configurations(filepath, chunk_sizes):
    """
    Test multiple chunk_size configurations and calculate average times.
    
    Args:
        filepath: Path to the CSV file to process
        chunk_sizes: List of chunk_size values to test
    
    Returns:
        Dictionary mapping chunk_size to average execution time
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    print(f"{'Chunk Size':<15} {'Run 1':>10} {'Run 2':>10} {'Run 3':>10} {'Average':>10}")
    print(f"{'-'*60}")
    
    for chunk_size in chunk_sizes:
        times = []
        
        for run in range(3):
            exec_time = run_benchmark(filepath, chunk_size)
            times.append(exec_time)
        
        avg_time = sum(times) / len(times)
        results[chunk_size] = avg_time
        
        print(f"{chunk_size:<15} {times[0]:>10.3f} {times[1]:>10.3f} {times[2]:>10.3f} {avg_time:>10.3f}")
    
    print(f"{'='*60}\n")
    
    return results

def main():
    """
    Main function to run benchmarks from command line.
    """
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <filepath> [start_chunk_size] [end_chunk_size]")
        print("Example: python benchmark.py transactions_small.csv 10 1000")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    
    # Default chunk sizes to test
    default_chunk_sizes = [1, 10, 50, 100, 200, 500, 1000, 5000]
    
    # Parse optional chunk size range
    if len(sys.argv) >= 4:
        start_chunk = int(sys.argv[2])
        end_chunk = int(sys.argv[3])
        
        # Generate range of chunk sizes
        chunk_sizes = []
        current = start_chunk
        while current <= end_chunk:
            chunk_sizes.append(current)
            if current < 100:
                current += 10
            elif current < 1000:
                current += 100
            else:
                current += 1000
    else:
        chunk_sizes = default_chunk_sizes
    
    # Run benchmarks
    results = test_configurations(filepath, chunk_sizes)
    
    # Find best configuration
    best_chunk_size = min(results, key=results.get)
    best_time = results[best_chunk_size]
    
    print(f"{'='*60}")
    print(f"BEST CONFIGURATION:")
    print(f"  Chunk Size: {best_chunk_size}")
    print(f"  Average Time: {best_time:.3f} seconds")
    print(f"{'='*60}\n")
    
    # Save results to JSON
    results_file = f"benchmark_results_{os.path.basename(filepath)}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'filepath': filepath,
            'results': {str(k): v for k, v in results.items()},
            'best_chunk_size': best_chunk_size,
            'best_time': best_time
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()