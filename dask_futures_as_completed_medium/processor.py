#!/usr/bin/env python3

import os
import json
import time
import random
from pathlib import Path
from dask.distributed import Client, as_completed

def process_file(filepath):
    """
    Process a single data file with simulated variable processing time.
    
    Args:
        filepath: Path to the CSV file to process
        
    Returns:
        Dictionary with processing results
    """
    filename = os.path.basename(filepath)
    
    # Simulate variable processing time (1-5 seconds)
    processing_time = random.uniform(1, 5)
    time.sleep(processing_time)
    
    # Simulate random failures (20% failure rate)
    if random.random() < 0.2:
        raise Exception(f"Processing failed for {filename}")
    
    # Simulate successful processing
    return {
        'filename': filename,
        'status': 'success',
        'processing_time': processing_time
    }

def main():
    """
    Main function to process all data files asynchronously.
    """
    # Create output directory if it doesn't exist
    output_dir = Path('/workspace/solution')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Dask client
    client = Client(n_workers=4, threads_per_worker=1, processes=True)
    
    try:
        # Get list of input files
        input_dir = Path('/workspace/data_pipeline/input')
        input_files = sorted(input_dir.glob('data_*.csv'))
        
        print(f"Found {len(input_files)} files to process")
        
        # Submit all tasks to the cluster
        futures = []
        for filepath in input_files:
            future = client.submit(process_file, str(filepath))
            futures.append(future)
        
        # Track results
        total_processed = 0
        successful = 0
        failed = 0
        completion_order = []
        
        # Process results as they complete using as_completed
        for future in as_completed(futures):
            total_processed += 1
            try:
                result = future.result()
                successful += 1
                completion_order.append(result['filename'])
                print(f"✓ Completed ({total_processed}/{len(input_files)}): {result['filename']}")
            except Exception as e:
                failed += 1
                # Extract filename from exception or use placeholder
                error_msg = str(e)
                if 'data_' in error_msg:
                    filename = error_msg.split('data_')[1].split('.csv')[0]
                    filename = f"data_{filename}.csv"
                else:
                    filename = f"unknown_file_{failed}"
                completion_order.append(filename)
                print(f"✗ Failed ({total_processed}/{len(input_files)}): {filename} - {e}")
        
        # Prepare results
        results = {
            'total_processed': total_processed,
            'successful': successful,
            'failed': failed,
            'completion_order': completion_order
        }
        
        # Write results to JSON file
        output_file = output_dir / 'results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Total processed: {total_processed}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {output_file}")
        
    finally:
        # Clean up
        client.close()

if __name__ == '__main__':
    main()