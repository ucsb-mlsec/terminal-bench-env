#!/usr/bin/env python3

import json
import multiprocessing
import os
import glob
import sys

# Global list that accumulates data in workers - memory leak!
worker_cache = []

def process_batch(batch_file):
    """
    Process a single batch file of sensor readings.
    This function has memory leak issues - accumulates data in worker_cache.
    """
    try:
        with open(batch_file, 'r') as f:
            readings = json.load(f)
        
        # Memory leak: accumulate readings in global list that never clears
        worker_cache.extend(readings)
        
        valid_count = 0
        temp_sum = 0
        humidity_sum = 0
        pressure_sum = 0
        
        for reading in readings:
            # Validate readings
            temp = reading.get('temperature', 0)
            humidity = reading.get('humidity', 0)
            pressure = reading.get('pressure', 0)
            
            is_valid = True
            if not (-50 <= temp <= 100):
                is_valid = False
            if not (0 <= humidity <= 100):
                is_valid = False
            if not (800 <= pressure <= 1200):
                is_valid = False
            
            if is_valid:
                valid_count += 1
                temp_sum += temp
                humidity_sum += humidity
                pressure_sum += pressure
        
        reading_count = len(readings)
        
        result = {
            'batch_file': os.path.basename(batch_file),
            'reading_count': reading_count,
            'valid_count': valid_count,
            'avg_temperature': temp_sum / valid_count if valid_count > 0 else 0,
            'avg_humidity': humidity_sum / valid_count if valid_count > 0 else 0,
            'avg_pressure': pressure_sum / valid_count if valid_count > 0 else 0
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {batch_file}: {e}", file=sys.stderr)
        return None

def main():
    """
    Main processing function with memory leak issues.
    """
    # Get all batch files
    batch_files = sorted(glob.glob('/data/sensor_batches/*.json'))
    
    if not batch_files:
        print("No batch files found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(batch_files)} batch files to process")
    
    # Memory leak: Pool created WITHOUT maxtasksperchild parameter
    # Workers never get recycled and accumulate memory
    pool = multiprocessing.Pool(processes=4)
    
    # Process all batches - memory accumulates in workers
    results = pool.map(process_batch, batch_files)
    
    # Filter out None results (errors)
    results = [r for r in results if r is not None]
    
    # Calculate summary statistics
    batches_processed = len(results)
    validation_errors = sum(r['reading_count'] - r['valid_count'] for r in results)
    
    # Prepare output
    output = {
        'batches_processed': batches_processed,
        'validation_errors': validation_errors,
        'peak_memory_mb': 0  # Not tracking in buggy version
    }
    
    # Write results
    os.makedirs('/workspace/pipeline', exist_ok=True)
    with open('/workspace/pipeline/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Processed {batches_processed} batches")
    print(f"Validation errors: {validation_errors}")
    
    # Memory leak: Pool not properly closed/joined
    # pool.close()
    # pool.join()

if __name__ == '__main__':
    main()