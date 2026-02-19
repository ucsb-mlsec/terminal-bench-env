#!/usr/bin/env python3

import os
import csv
import time
import random
import multiprocessing
from multiprocessing import Process, Lock, Queue, Manager
from pathlib import Path
import json

# Fixed random seed for deterministic behavior
random.seed(42)

# Global paths
DATA_DIR = "/opt/worker_app/data/"
OUTPUT_DIR = "/opt/worker_app/output/"
NUM_WORKERS = 4

def setup_directories():
    """Create necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sample_data():
    """Generate sample CSV files if they don't exist"""
    setup_directories()
    
    # Generate 20 sample CSV files
    for i in range(20):
        filepath = os.path.join(DATA_DIR, f"data_{i:02d}.csv")
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'value', 'category'])
                for j in range(100):
                    writer.writerow([j, random.randint(1, 1000), f"cat_{j % 5}"])

def process_file(filepath, output_lock, stats_lock, wait_times):
    """Process a single data file"""
    filename = os.path.basename(filepath)
    
    # Read and process data
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'id': row['id'],
                'value': int(row['value']),
                'category': row['category']
            })
    
    # Simulate some processing
    time.sleep(0.1)
    
    # Calculate statistics
    total = sum(item['value'] for item in data)
    avg = total / len(data) if data else 0
    max_val = max((item['value'] for item in data), default=0)
    min_val = min((item['value'] for item in data), default=0)
    
    # Group by category
    categories = {}
    for item in data:
        cat = item['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item['value'])
    
    category_stats = {}
    for cat, values in categories.items():
        category_stats[cat] = {
            'count': len(values),
            'sum': sum(values),
            'avg': sum(values) / len(values)
        }
    
    # This is the BOTTLENECK: Writing output with a lock held for a long time
    # Each worker must wait for this lock, and we're doing slow operations while holding it
    wait_start = time.time()
    output_lock.acquire()
    wait_time = time.time() - wait_start
    
    try:
        # Simulate slow I/O operation while holding the lock (the bottleneck!)
        time.sleep(0.8)  # This causes significant contention
        
        output_file = os.path.join(OUTPUT_DIR, f"result_{filename}")
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['total', total])
            writer.writerow(['average', f"{avg:.2f}"])
            writer.writerow(['max', max_val])
            writer.writerow(['min', min_val])
            writer.writerow(['', ''])
            writer.writerow(['category_stats', ''])
            for cat, stats in category_stats.items():
                writer.writerow([cat, f"count={stats['count']}, avg={stats['avg']:.2f}"])
    finally:
        output_lock.release()
    
    # Track wait time
    stats_lock.acquire()
    try:
        wait_times.append(wait_time)
    finally:
        stats_lock.release()
    
    return filename

def worker(task_queue, output_lock, stats_lock, wait_times):
    """Worker process that processes files from the queue"""
    while True:
        try:
            filepath = task_queue.get(timeout=1)
            if filepath is None:  # Poison pill
                break
            process_file(filepath, output_lock, stats_lock, wait_times)
        except:
            break

def main():
    print("Starting data processing application...")
    print(f"Using {NUM_WORKERS} worker processes")
    
    # Setup and generate sample data
    setup_directories()
    generate_sample_data()
    
    # Get all CSV files
    data_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"Found {len(data_files)} data files to process")
    
    # Create synchronization primitives
    manager = Manager()
    task_queue = manager.Queue()
    output_lock = manager.Lock()  # THE BOTTLENECK
    stats_lock = manager.Lock()
    wait_times = manager.list()
    
    # Add files to queue
    for filepath in data_files:
        task_queue.put(filepath)
    
    # Add poison pills
    for _ in range(NUM_WORKERS):
        task_queue.put(None)
    
    # Start workers
    start_time = time.time()
    workers = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker, args=(task_queue, output_lock, stats_lock, wait_times))
        p.start()
        workers.append(p)
    
    # Wait for all workers to complete
    for p in workers:
        p.join()
    
    elapsed_time = time.time() - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Processed {len(data_files)} files")
    print(f"Results written to {OUTPUT_DIR}")
    
    # Calculate total wait time
    total_wait = sum(wait_times)
    print(f"\nPerformance Analysis:")
    print(f"Total wait time on output_lock: {total_wait:.1f} seconds")
    print(f"Average wait per file: {total_wait/len(data_files):.2f} seconds")
    
    # Write analysis to file
    analysis = {
        "bottleneck_name": "output_lock",
        "total_wait_seconds": round(total_wait, 1)
    }
    
    with open('/tmp/analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis written to /tmp/analysis.json")

if __name__ == "__main__":
    main()