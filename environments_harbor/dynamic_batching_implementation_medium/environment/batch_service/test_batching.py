#!/usr/bin/env python3

import sys
import time
import threading
from queue import Queue, Empty
from processor import process_single, process_batch
from request_generator import RequestGenerator

# Global statistics tracking
stats = {
    'total_batches': 0,
    'total_requests': 0,
    'batch_sizes': []
}

def batch_processor(request_queue, response_dict, shutdown_event):
    """
    Process requests in batches with timeout and size limits.
    """
    MAX_BATCH_SIZE = 20
    MAX_WAIT_TIME = 0.05  # 50 milliseconds
    
    while not shutdown_event.is_set() or not request_queue.empty():
        batch = []
        batch_ids = []
        start_time = time.time()
        
        # Collect requests up to max batch size or timeout
        while len(batch) < MAX_BATCH_SIZE:
            remaining_time = MAX_WAIT_TIME - (time.time() - start_time)
            
            if remaining_time <= 0 and len(batch) > 0:
                # Timeout reached with at least one request
                break
            
            try:
                # Wait for request with timeout
                timeout = max(0.001, remaining_time) if len(batch) > 0 else MAX_WAIT_TIME
                request_id, data = request_queue.get(timeout=timeout)
                batch.append(data)
                batch_ids.append(request_id)
            except Empty:
                if len(batch) > 0:
                    # Timeout with pending requests
                    break
                if shutdown_event.is_set():
                    # No more requests coming
                    break
                continue
        
        # Process the batch if we have any requests
        if batch:
            if len(batch) == 1:
                # Process single request
                result = [process_single(batch[0])]
            else:
                # Process as batch
                result = process_batch(batch)
            
            # Store results
            for req_id, res in zip(batch_ids, result):
                response_dict[req_id] = res
            
            # Update statistics
            stats['total_batches'] += 1
            stats['total_requests'] += len(batch)
            stats['batch_sizes'].append(len(batch))

def run_batch_server(requests, timeout=10):
    """
    Run the batch server with given requests.
    Returns response dictionary and statistics.
    """
    request_queue = Queue()
    response_dict = {}
    shutdown_event = threading.Event()
    
    # Start the batch processor thread
    processor_thread = threading.Thread(
        target=batch_processor,
        args=(request_queue, response_dict, shutdown_event)
    )
    processor_thread.start()
    
    # Submit all requests
    for req_id, data in requests:
        request_queue.put((req_id, data))
    
    # Wait for all requests to be processed
    start_wait = time.time()
    while len(response_dict) < len(requests):
        if time.time() - start_wait > timeout:
            print(f"Timeout: Only {len(response_dict)}/{len(requests)} requests processed")
            break
        time.sleep(0.001)
    
    # Signal shutdown and wait for thread
    shutdown_event.set()
    processor_thread.join(timeout=1)
    
    return response_dict, stats

if __name__ == "__main__":
    print("Batch Server - Testing Implementation")
    print("=" * 50)
    
    # Generate test requests
    print("Generating 100 test requests...")
    generator = RequestGenerator(seed=42)
    test_requests = generator.generate_requests(100)
    
    print(f"Generated {len(test_requests)} requests")
    print("\nStarting batch server...")
    
    # Reset statistics
    stats = {
        'total_batches': 0,
        'total_requests': 0,
        'batch_sizes': []
    }
    
    start_time = time.time()
    responses, final_stats = run_batch_server(test_requests, timeout=10)
    elapsed_time = time.time() - start_time
    
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Requests processed: {len(responses)}/{len(test_requests)}")
    print(f"Total batches: {final_stats['total_batches']}")
    
    if final_stats['batch_sizes']:
        avg_batch_size = sum(final_stats['batch_sizes']) / len(final_stats['batch_sizes'])
        print(f"Average batch size: {avg_batch_size:.1f}")
    else:
        avg_batch_size = 0.0
        print("No batches processed!")
    
    # Verify correctness
    print("\nVerifying results...")
    all_correct = True
    
    # Check all requests were processed
    if len(responses) != len(test_requests):
        print(f"ERROR: Expected {len(test_requests)} responses, got {len(responses)}")
        all_correct = False
    
    # Verify each response
    for req_id, data in test_requests:
        if req_id not in responses:
            print(f"ERROR: Missing response for request {req_id}")
            all_correct = False
            continue
        
        # Calculate expected result
        expected = process_single(data)
        actual = responses[req_id]
        
        if expected != actual:
            print(f"ERROR: Request {req_id}: expected {expected}, got {actual}")
            all_correct = False
    
    # Check that batching is actually occurring
    if final_stats['total_batches'] >= len(test_requests):
        print("WARNING: No batching detected (processing one-by-one)")
        all_correct = False
    
    print("\n" + "=" * 50)
    if all_correct:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    # Save results
    results_file = "/workspace/batch_service/results.txt"
    with open(results_file, 'w') as f:
        f.write(f"total_batches={final_stats['total_batches']}\n")
        f.write(f"avg_batch_size={avg_batch_size:.1f}\n")
        f.write(f"all_correct={'true' if all_correct else 'false'}\n")
    
    print(f"\nResults saved to {results_file}")
    
    sys.exit(0 if all_correct else 1)