#!/usr/bin/env python3

import multiprocessing
from multiprocessing import Process, Queue
import time
import random

def data_generator(queue, count):
    """Generate sensor readings and put them in the queue"""
    for i in range(count):
        reading = {
            'id': i,
            'temperature': random.uniform(20.0, 30.0),
            'pressure': random.uniform(1000.0, 1020.0),
            'timestamp': time.time()
        }
        queue.put(reading)  # This will block when queue is full
    queue.put(None)  # Sentinel value to signal completion

def worker_process(input_queue, output_queue):
    """Transform sensor data from Celsius to Fahrenheit"""
    while True:
        item = input_queue.get()  # Blocking get from input
        if item is None:
            output_queue.put(None)  # Signal downstream that this worker is done
            break
        
        # Transform the data
        transformed = item.copy()
        transformed['temperature'] = (item['temperature'] * 9/5) + 32
        
        output_queue.put(transformed)  # This will block when output queue is full

def collector_process(queue, result_list):
    """Collect processed items from the queue"""
    workers_done = 0
    total_workers = 4
    
    while workers_done < total_workers:
        item = queue.get()
        if item is None:
            workers_done += 1
        else:
            result_list.append(item)

def run_pipeline(item_count):
    """Main pipeline that orchestrates the data processing"""
    # Create bounded queues with limited capacity
    input_queue = Queue(maxsize=100)
    output_queue = Queue(maxsize=100)
    
    # Shared result list
    manager = multiprocessing.Manager()
    results = manager.list()
    
    # Start the generator process
    generator = Process(target=data_generator, args=(input_queue, item_count))
    generator.start()
    
    # Start worker processes
    workers = []
    for _ in range(4):
        worker = Process(target=worker_process, args=(input_queue, output_queue))
        worker.start()
        workers.append(worker)
    
    # Start collector process
    collector = Process(target=collector_process, args=(output_queue, results))
    collector.start()
    
    # Wait for all processes to complete
    generator.join()  # Wait for generator to finish
    
    # Put sentinel values for each worker
    for _ in range(4):
        input_queue.put(None)
    
    for worker in workers:
        worker.join()  # Wait for workers to finish
    
    collector.join()  # Wait for collector to finish
    
    return list(results)

if __name__ == '__main__':
    print("Testing pipeline with small dataset (100 items)...")
    start = time.time()
    results = run_pipeline(100)
    elapsed = time.time() - start
    print(f"Small test completed: {len(results)} items processed in {elapsed:.2f} seconds")
    
    print("\nTesting pipeline with large dataset (50000 items)...")
    print("WARNING: This will likely deadlock due to queue handling issues...")
    start = time.time()
    results = run_pipeline(50000)
    elapsed = time.time() - start
    print(f"SUCCESS: Processed {len(results)} items in {elapsed:.2f} seconds")