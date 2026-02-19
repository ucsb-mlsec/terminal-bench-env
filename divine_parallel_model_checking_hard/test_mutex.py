#!/usr/bin/env python3

import threading
import time
from mutex_system import MutexSystem

def worker(mutex_system, num_increments, worker_id):
    """Worker thread that performs increments on the shared counter."""
    for i in range(num_increments):
        mutex_system.increment()
    print(f"Worker {worker_id} completed {num_increments} increments")

def main():
    # Test configuration
    num_threads = 10
    increments_per_thread = 100
    
    print(f"Starting mutex stress test...")
    print(f"Threads: {num_threads}")
    print(f"Increments per thread: {increments_per_thread}")
    print(f"Expected final count: {num_threads * increments_per_thread}")
    print("-" * 50)
    
    # Create the shared mutex system
    mutex_system = MutexSystem()
    
    # Create threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(mutex_system, increments_per_thread, i))
        threads.append(thread)
    
    # Start all threads
    print("Starting all threads...")
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    print("Waiting for threads to complete...")
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    # Get final counter value
    actual_count = mutex_system.get_count()
    expected_count = num_threads * increments_per_thread
    
    print("-" * 50)
    print(f"Test completed in {end_time - start_time:.2f} seconds")
    print(f"Expected count: {expected_count}")
    print(f"Actual count: {actual_count}")
    
    if actual_count == expected_count:
        print("✓ PASS: Mutex works correctly - no race conditions detected")
    else:
        print(f"✗ FAIL: Race condition detected - lost {expected_count - actual_count} increments")

if __name__ == "__main__":
    main()