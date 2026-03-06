#!/usr/bin/env python3

import time
import random
from typing import Iterator, Dict, Any

def generate_requests(num_requests: int = 100) -> Iterator[Dict[str, Any]]:
    """
    Generate requests with realistic timing patterns including bursts and sparse periods.
    
    Args:
        num_requests: Total number of requests to generate
        
    Yields:
        Dictionary with 'id' and 'data' fields
    """
    for i in range(num_requests):
        # Create request
        request = {
            'id': i,
            'data': random.randint(1, 100)
        }
        
        yield request
        
        # Don't sleep after the last request
        if i < num_requests - 1:
            # Determine inter-arrival time based on patterns
            # 40% of the time: burst pattern (very close together)
            # 30% of the time: moderate spacing
            # 30% of the time: sparse pattern (larger gaps)
            
            rand_val = random.random()
            
            if rand_val < 0.4:
                # Burst: 0-10ms between requests
                sleep_time = random.uniform(0.001, 0.010)
            elif rand_val < 0.7:
                # Moderate: 10-50ms between requests
                sleep_time = random.uniform(0.010, 0.050)
            else:
                # Sparse: 100-200ms between requests
                sleep_time = random.uniform(0.100, 0.200)
            
            time.sleep(sleep_time)


def generate_bursty_requests(num_requests: int = 100, burst_size: int = 10) -> Iterator[Dict[str, Any]]:
    """
    Generate requests in distinct bursts followed by gaps.
    
    Args:
        num_requests: Total number of requests to generate
        burst_size: Approximate number of requests per burst
        
    Yields:
        Dictionary with 'id' and 'data' fields
    """
    request_id = 0
    
    while request_id < num_requests:
        # Generate a burst
        current_burst_size = min(random.randint(5, burst_size), num_requests - request_id)
        
        for _ in range(current_burst_size):
            request = {
                'id': request_id,
                'data': random.randint(1, 100)
            }
            yield request
            request_id += 1
            
            # Very short delay within burst
            if request_id < num_requests:
                time.sleep(random.uniform(0.001, 0.005))
        
        # Gap between bursts
        if request_id < num_requests:
            time.sleep(random.uniform(0.100, 0.200))


if __name__ == '__main__':
    print("Generating sample requests...")
    for req in generate_requests(20):
        print(f"Request {req['id']}: data={req['data']}")