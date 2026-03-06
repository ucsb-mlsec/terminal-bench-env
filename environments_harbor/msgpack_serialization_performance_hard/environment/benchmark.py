#!/usr/bin/env python3

import time
import json
import msgpack
import subprocess
import sys
import os

def validate_output(msgpack_file, expected_count=10000):
    """Validate that the MessagePack file contains all events with correct structure."""
    try:
        with open(msgpack_file, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False)
        
        if not isinstance(data, list):
            print(f"Error: Expected list, got {type(data)}")
            return False
        
        if len(data) != expected_count:
            print(f"Error: Expected {expected_count} events, got {len(data)}")
            return False
        
        # Validate first event has expected structure
        if data:
            sample = data[0]
            if not isinstance(sample, dict):
                print(f"Error: Expected dict for event, got {type(sample)}")
                return False
        
        return True
    except Exception as e:
        print(f"Error validating output: {e}")
        return False

def run_processor():
    """Run the processor.py script and measure its execution time."""
    try:
        start_time = time.perf_counter()
        result = subprocess.run(
            [sys.executable, 'processor.py'],
            cwd='/workspace/event_processor',
            capture_output=True,
            text=True,
            timeout=300
        )
        end_time = time.perf_counter()
        
        if result.returncode != 0:
            print(f"Error running processor.py:")
            print(result.stderr)
            return None
        
        elapsed = end_time - start_time
        return elapsed
    except subprocess.TimeoutExpired:
        print("Error: Processor timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"Error running processor: {e}")
        return None

def main():
    print("=" * 60)
    print("Event Processor Performance Benchmark")
    print("=" * 60)
    
    # Check if required files exist
    input_file = '/workspace/event_processor/events.json'
    output_file = '/workspace/event_processor/events.msgpack'
    processor_file = '/workspace/event_processor/processor.py'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if not os.path.exists(processor_file):
        print(f"Error: Processor script not found: {processor_file}")
        sys.exit(1)
    
    # Load input to get expected count
    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        expected_count = len(input_data)
        print(f"Input file contains {expected_count} events")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Remove old output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("\nRunning baseline measurement...")
    baseline_time = run_processor()
    
    if baseline_time is None:
        print("Failed to run processor")
        sys.exit(1)
    
    print(f"Baseline processing time: {baseline_time:.2f} seconds")
    
    # Validate output
    print("\nValidating output...")
    if not os.path.exists(output_file):
        print(f"Error: Output file not created: {output_file}")
        sys.exit(1)
    
    if not validate_output(output_file, expected_count):
        print("Output validation failed")
        sys.exit(1)
    
    print(f"Output validated successfully: {expected_count} events")
    print("=" * 60)
    
    return baseline_time

if __name__ == '__main__':
    baseline = main()