#!/usr/bin/env python3

import multiprocessing
import time
import sys
import random
from token_manager import TokenManager
from api_client import APIClient


def worker_task(worker_id):
    """
    Simulates a worker that makes multiple API calls.
    Returns True if all calls succeed, False otherwise.
    """
    print(f"[Worker {worker_id}] Starting...")
    
    try:
        # Create token manager and API client instances
        token_manager = TokenManager('data/tokens.json')
        api_client = APIClient(token_manager)
        
        # Track successes and failures
        call_count = 4
        success_count = 0
        
        for call_num in range(call_count):
            try:
                print(f"[Worker {worker_id}] Making API call {call_num + 1}/{call_count}")
                
                # Make API call
                result = api_client.fetch_data()
                
                if result:
                    success_count += 1
                    print(f"[Worker {worker_id}] Call {call_num + 1} succeeded")
                else:
                    print(f"[Worker {worker_id}] Call {call_num + 1} failed")
                
                # Add small random delay between calls
                time.sleep(random.uniform(0.05, 0.1))
                
            except Exception as e:
                print(f"[Worker {worker_id}] Call {call_num + 1} raised exception: {e}")
        
        # Worker succeeds if all calls succeeded
        worker_success = (success_count == call_count)
        
        if worker_success:
            print(f"[Worker {worker_id}] Completed successfully ({success_count}/{call_count} calls)")
        else:
            print(f"[Worker {worker_id}] Failed ({success_count}/{call_count} calls)")
        
        return worker_success
        
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        return False


def main():
    """
    Main execution: runs 5 workers concurrently and reports results.
    """
    print("Starting concurrent worker test with 5 workers...\n")
    
    num_workers = 5
    worker_ids = range(1, num_workers + 1)
    
    # Create a process pool and run workers concurrently
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(worker_task, worker_ids)
    
    # Count successful workers
    successful_workers = sum(results)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successful_workers} out of {num_workers} workers succeeded")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    if successful_workers == num_workers:
        print("✓ All workers succeeded!")
        sys.exit(0)
    else:
        print(f"✗ {num_workers - successful_workers} worker(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()