#!/usr/bin/env python3

import threading
import time
import random
from queue import Queue

# Global shared data structures
collected_data = []
processed_data = []
final_result = {'total_readings': 0, 'final_sum': 0}

def collect_data(worker_id, start, end):
    """Collect sensor readings from start to end (inclusive)"""
    count = 0
    for i in range(start, end + 1):
        # Simulate sensor reading delay
        time.sleep(random.uniform(0.01, 0.05))
        collected_data.append(i)
        count += 1
    print(f'Worker {worker_id} collected {count} readings')
    # BUG: Does NOT properly signal completion

def process_data(worker_id, data_chunk):
    """Process data chunk by applying transformation"""
    # BUG: Does NOT wait for data collection to complete before starting
    processed_count = 0
    for item in data_chunk:
        # Simulate processing delay
        time.sleep(random.uniform(0.01, 0.03))
        # Identity operation (multiply by 1)
        processed_item = item * 1
        processed_data.append(processed_item)
        processed_count += 1
    print(f'Processing worker {worker_id} processed {processed_count} items')

def aggregate_results():
    """Aggregate all processed results"""
    # BUG: Does NOT wait for processing to complete before starting
    total_sum = sum(processed_data)
    total_count = len(processed_data)
    final_result['total_readings'] = total_count
    final_result['final_sum'] = total_sum
    print('Aggregation complete')

def main():
    """Main pipeline execution"""
    print("Starting data processing pipeline...")
    
    # Initialize sensor readings (1 to 100)
    total_readings = 100
    
    # Stage 1: Create collection worker threads
    collection_threads = []
    readings_per_worker = 25
    
    # Create 4 collection workers
    for i in range(4):
        start = i * readings_per_worker + 1
        end = (i + 1) * readings_per_worker
        thread = threading.Thread(target=collect_data, args=(i, start, end))
        collection_threads.append(thread)
    
    # Start all collection threads immediately
    for thread in collection_threads:
        thread.start()
    
    # BUG: Creates processing threads WITHOUT waiting for collection to complete
    processing_threads = []
    
    # Attempt to divide collected_data into chunks
    # BUG: Data may not be ready yet!
    time.sleep(0.1)  # Inadequate wait time
    
    # Create 3 processing workers
    chunk_size = len(collected_data) // 3 if len(collected_data) > 0 else 0
    for i in range(3):
        start_idx = i * chunk_size
        if i == 2:
            # Last worker gets remaining items
            data_chunk = collected_data[start_idx:]
        else:
            end_idx = (i + 1) * chunk_size
            data_chunk = collected_data[start_idx:end_idx]
        
        thread = threading.Thread(target=process_data, args=(i, data_chunk))
        processing_threads.append(thread)
    
    # Start processing threads immediately
    for thread in processing_threads:
        thread.start()
    
    # BUG: Creates aggregation thread WITHOUT waiting for processing to complete
    time.sleep(0.1)  # Inadequate wait time
    aggregation_thread = threading.Thread(target=aggregate_results)
    aggregation_thread.start()
    
    # Poor thread joins - may not wait for all to complete
    aggregation_thread.join(timeout=1)
    
    print(f"Pipeline complete. Results: {final_result}")
    return final_result

if __name__ == '__main__':
    result = main()
    print(f"\nFinal Results:")
    print(f"Total readings: {result['total_readings']}")
    print(f"Final sum: {result['final_sum']}")
    print(f"Expected sum: 5050")
