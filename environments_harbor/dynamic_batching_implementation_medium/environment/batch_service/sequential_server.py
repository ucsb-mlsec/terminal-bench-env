#!/usr/bin/env python3

import time
import sys
from processor import Processor
from request_generator import RequestGenerator

def sequential_server():
    """Process requests one at a time without batching."""
    processor = Processor()
    generator = RequestGenerator()
    
    print("Starting sequential server (processing one request at a time)...")
    print("-" * 60)
    
    request_count = 0
    start_time = time.time()
    
    # Process each request individually as it arrives
    for request_id, data in generator.generate_requests(count=100):
        request_start = time.time()
        
        # Process single item
        result = processor.process([data])[0]
        
        process_time = (time.time() - request_start) * 1000
        request_count += 1
        
        print(f"Request {request_id}: Processed individually in {process_time:.2f}ms, Result: {result}")
        
        # Simulate waiting for next request
        time.sleep(0.001)
    
    total_time = time.time() - start_time
    
    print("-" * 60)
    print(f"Sequential processing complete:")
    print(f"  Total requests: {request_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per request: {(total_time/request_count)*1000:.2f}ms")
    print(f"  Batches used: {request_count} (one per request - inefficient!)")

if __name__ == "__main__":
    sequential_server()