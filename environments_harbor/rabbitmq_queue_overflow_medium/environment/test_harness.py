#!/usr/bin/env python3

import queue
import time
import sys
import importlib
import os
import traceback

def main():
    # Create a queue and populate it with 20,000 messages
    message_queue = queue.Queue()
    total_messages = 20000
    
    print(f"Populating queue with {total_messages} messages...")
    for i in range(total_messages):
        message_queue.put({'id': i})
    
    print(f"Queue populated. Starting consumer test...")
    
    # Try to import fixed_consumer, fall back to broken_consumer
    consumer_module = None
    module_name = None
    
    if os.path.exists('/home/user/fixed_consumer.py'):
        module_name = 'fixed_consumer'
        print("Using fixed_consumer.py")
    elif os.path.exists('/home/user/broken_consumer.py'):
        module_name = 'broken_consumer'
        print("Using broken_consumer.py")
    else:
        print("ERROR: No consumer module found!")
        write_results(0, 0.0, 'CRASH')
        sys.exit(1)
    
    try:
        # Add /home/user to path if not already there
        if '/home/user' not in sys.path:
            sys.path.insert(0, '/home/user')
        
        consumer_module = importlib.import_module(module_name)
        
        if not hasattr(consumer_module, 'process_messages'):
            print("ERROR: Consumer module missing process_messages function!")
            write_results(0, 0.0, 'CRASH')
            sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: Failed to import consumer module: {e}")
        traceback.print_exc()
        write_results(0, 0.0, 'CRASH')
        sys.exit(1)
    
    # Measure execution time and process messages
    start_time = time.time()
    processed_count = 0
    status = 'CRASH'
    
    try:
        # Call the consumer's process_messages function
        result = consumer_module.process_messages(message_queue, total_messages)
        
        # Calculate how many messages were actually processed
        processed_count = total_messages - message_queue.qsize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Processing completed!")
        print(f"Messages processed: {processed_count}/{total_messages}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        
        # Determine status
        if processed_count == total_messages:
            if elapsed_time < 300:  # Must complete in under 300 seconds
                processing_rate = processed_count / elapsed_time
                print(f"Processing rate: {processing_rate:.2f} messages/second")
                if processing_rate >= 100:
                    status = 'PASS'
                    print("✓ Test PASSED!")
                else:
                    status = 'FAIL'
                    print(f"✗ Test FAILED: Processing rate too slow ({processing_rate:.2f} < 100 msg/s)")
            else:
                status = 'FAIL'
                print(f"✗ Test FAILED: Timeout (took {elapsed_time:.2f}s > 300s)")
        else:
            status = 'FAIL'
            print(f"✗ Test FAILED: Incomplete processing ({processed_count}/{total_messages})")
        
        write_results(processed_count, elapsed_time, status)
        
    except MemoryError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        processed_count = total_messages - message_queue.qsize()
        print(f"✗ CRASH: Memory error after processing {processed_count} messages")
        print(f"Error: {e}")
        write_results(processed_count, elapsed_time, 'CRASH')
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        processed_count = total_messages - message_queue.qsize()
        print(f"✗ CRASH: Exception after processing {processed_count} messages")
        print(f"Error: {e}")
        traceback.print_exc()
        write_results(processed_count, elapsed_time, 'CRASH')

def write_results(processed, elapsed_time, status):
    """Write results to the results file"""
    try:
        with open('/home/user/results.txt', 'w') as f:
            f.write(f"PROCESSED={processed}\n")
            f.write(f"TIME={elapsed_time:.2f}\n")
            f.write(f"STATUS={status}\n")
        print(f"\nResults written to /home/user/results.txt")
    except Exception as e:
        print(f"ERROR: Failed to write results file: {e}")

if __name__ == '__main__':
    main()