#!/usr/bin/env python3

import sys
import time
import gc
import psutil
from notifier import EventDispatcher, Subscriber

def get_memory_mb():
    """Return current process memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def main():
    print("=" * 60)
    print("Event System Memory Leak Test")
    print("=" * 60)
    
    # Force garbage collection and get baseline
    gc.collect()
    time.sleep(0.1)
    initial_memory = get_memory_mb()
    print(f"Initial memory: {initial_memory:.2f} MB\n")
    
    # Create event dispatcher
    dispatcher = EventDispatcher()
    
    # Test parameters
    total_iterations = 1500
    subscribers_per_iteration = 15
    events_per_iteration = 7
    report_interval = 100
    
    print("Starting test...")
    print(f"Iterations: {total_iterations}")
    print(f"Subscribers per iteration: {subscribers_per_iteration}")
    print(f"Events per iteration: {events_per_iteration}")
    print()
    
    start_time = time.time()
    
    for iteration in range(total_iterations):
        # Create multiple subscribers
        subscribers = []
        for i in range(subscribers_per_iteration):
            sub = Subscriber(f"sub_{iteration}_{i}")
            subscribers.append(sub)
            
            # Subscribe to multiple event types
            sub.subscribe(dispatcher, "user_login")
            sub.subscribe(dispatcher, "user_logout")
            sub.subscribe(dispatcher, "data_update")
        
        # Dispatch some events
        for j in range(events_per_iteration):
            dispatcher.dispatch("user_login", {"user_id": j})
            dispatcher.dispatch("user_logout", {"user_id": j})
            dispatcher.dispatch("data_update", {"data": f"test_{j}"})
        
        # Clean up subscribers
        for sub in subscribers:
            sub.unsubscribe_all()
        
        # Delete references
        del subscribers
        
        # Periodic reporting and garbage collection
        if (iteration + 1) % report_interval == 0:
            gc.collect()
            time.sleep(0.01)
            current_memory = get_memory_mb()
            growth = current_memory - initial_memory
            print(f"Iteration {iteration + 1:4d}: Memory: {current_memory:.2f} MB (growth: +{growth:.2f} MB)")
        
        # Small delay to simulate realistic usage
        time.sleep(0.002)
    
    # Final garbage collection
    gc.collect()
    time.sleep(0.1)
    final_memory = get_memory_mb()
    memory_growth = final_memory - initial_memory
    
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Test duration: {elapsed_time:.1f} seconds")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory growth: {memory_growth:.2f} MB")
    print()
    
    # Determine if memory leak exists
    leak_threshold = 5.0  # MB
    
    if memory_growth > leak_threshold:
        print(f"Memory leak detected: YES (growth > {leak_threshold} MB)")
        print("Memory stable: NO")
        return 1
    else:
        print(f"Memory leak detected: NO (growth <= {leak_threshold} MB)")
        print("Memory stable: YES")
        return 0

if __name__ == "__main__":
    sys.exit(main())