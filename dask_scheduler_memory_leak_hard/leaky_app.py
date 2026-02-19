#!/usr/bin/env python3

from dask.distributed import Client, LocalCluster
import time
import psutil
import os
import sys

def get_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def compute_sum_of_squares(n):
    """Compute sum of squares up to n"""
    result = sum(i * i for i in range(n))
    return result

def process_data(data_size):
    """Simulate data processing by creating and manipulating a list"""
    data = list(range(data_size))
    processed = [x * 2 + 1 for x in data]
    return sum(processed)

def complex_computation(x, y):
    """Perform a more complex computation combining multiple operations"""
    temp = compute_sum_of_squares(x)
    return temp + process_data(y)

if __name__ == '__main__':
    print("Starting Dask distributed application...")
    print("Initializing cluster with 2 workers...")
    
    # Create a local cluster with resource limits
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        memory_limit='500MB',
        silence_logs=False
    )
    
    # Connect client to the cluster
    client = Client(cluster)
    
    print(f"Cluster dashboard available at: {client.dashboard_link}")
    print(f"Initial memory usage: {get_memory_mb():.1f} MB")
    print("\nStarting task submission loop...")
    print("Press Ctrl+C to stop\n")
    
    # This list will accumulate all results, causing a memory leak
    all_results_history = []
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Submit a batch of computational tasks
            futures = []
            
            # Submit first type of tasks
            for i in range(50):
                future = client.submit(compute_sum_of_squares, (i % 20) + 100)
                futures.append(future)
            
            # Submit second type of tasks
            for i in range(30):
                future = client.submit(process_data, (i % 15) + 200)
                futures.append(future)
            
            # Submit combined tasks
            for i in range(20):
                future = client.submit(complex_computation, (i % 10) + 50, (i % 8) + 100)
                futures.append(future)
            
            # Gather all results from the futures
            results = client.gather(futures)
            
            # Store results in history - THIS CAUSES THE MEMORY LEAK
            # These results accumulate indefinitely and are never cleared
            all_results_history.append(results)
            
            # Optional: compute some statistics on current batch
            batch_sum = sum(results)
            batch_avg = batch_sum / len(results)
            
            # Print progress every 10 iterations
            if iteration % 10 == 0:
                current_memory = get_memory_mb()
                print(f"Iteration {iteration:4d} | Memory: {current_memory:7.1f} MB | "
                      f"Batch avg: {batch_avg:10.1f} | Total batches stored: {len(all_results_history)}")
            
            # Small delay between iterations
            time.sleep(0.15)
            
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        print(f"Final memory usage: {get_memory_mb():.1f} MB")
        print(f"Total iterations completed: {iteration}")
        print(f"Total result batches accumulated: {len(all_results_history)}")
        
    finally:
        # Clean shutdown
        client.close()
        cluster.close()
        print("Cluster closed. Exiting.")