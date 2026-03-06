#!/usr/bin/env python3

import asyncio
import sys
import os
import time
import re
from pathlib import Path

async def run_app_and_monitor():
    """Run the app.py and monitor its output for queue size information."""
    max_queue_size = 0
    test_passed = False
    crashed = False
    
    try:
        # Import the app module
        sys.path.insert(0, '/home/agent/message_processor')
        import app
        
        # Create a custom queue monitor
        original_qsize = None
        queue_sizes = []
        
        # Run the app with monitoring
        start_time = time.time()
        test_duration = 30
        
        # Start the app in background
        app_task = asyncio.create_task(app.main())
        
        # Monitor the queue size
        while time.time() - start_time < test_duration:
            try:
                if hasattr(app, 'message_queue') and app.message_queue is not None:
                    current_size = app.message_queue.qsize()
                    queue_sizes.append(current_size)
                    if current_size > max_queue_size:
                        max_queue_size = current_size
                await asyncio.sleep(0.1)  # Check every 100ms
            except Exception as e:
                print(f"Error monitoring queue: {e}")
                crashed = True
                break
        
        # Cancel the app task after test duration
        app_task.cancel()
        try:
            await app_task
        except asyncio.CancelledError:
            pass
        
        # Check if test passed
        elapsed = time.time() - start_time
        if not crashed and elapsed >= test_duration - 1 and max_queue_size <= 500:
            test_passed = True
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        crashed = True
        test_passed = False
    
    return max_queue_size, test_passed

def main():
    """Main test function."""
    print("Starting message processor test...")
    print("Test duration: 30 seconds")
    print("Maximum allowed queue size: 500 messages")
    print("-" * 50)
    
    try:
        # Run the async test
        max_queue_size, test_passed = asyncio.run(run_app_and_monitor())
        
        print("-" * 50)
        print(f"Test completed")
        print(f"Maximum queue size observed: {max_queue_size}")
        print(f"Test passed: {test_passed}")
        
        # Write results to file
        result_path = '/home/agent/message_processor/result.txt'
        with open(result_path, 'w') as f:
            f.write(f"max_queue_size={max_queue_size}\n")
            f.write(f"test_passed={'true' if test_passed else 'false'}\n")
        
        print(f"\nResults written to {result_path}")
        
        # Exit with appropriate code
        sys.exit(0 if test_passed else 1)
        
    except Exception as e:
        print(f"Fatal error during test: {e}")
        # Write failure result
        result_path = '/home/agent/message_processor/result.txt'
        with open(result_path, 'w') as f:
            f.write(f"max_queue_size=0\n")
            f.write(f"test_passed=false\n")
        sys.exit(1)

if __name__ == "__main__":
    main()