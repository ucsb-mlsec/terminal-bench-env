#!/usr/bin/env python3

import json
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import sys
import os


def process_file(file_path):
    """
    Process a single JSON file and return statistics.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Perform some transformation - count records and sum values
        if isinstance(data, list):
            record_count = len(data)
            total_value = sum(item.get('value', 0) for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            record_count = len(data)
            total_value = sum(v for v in data.values() if isinstance(v, (int, float)))
        else:
            record_count = 0
            total_value = 0
        
        return {
            'file': os.path.basename(file_path),
            'record_count': record_count,
            'total_value': total_value,
            'status': 'success'
        }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'error': str(e),
            'status': 'failed'
        }


def main():
    # Create output directory if it doesn't exist
    output_dir = Path('/app/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files from data directory
    data_dir = Path('/app/data')
    json_files = list(data_dir.glob('*.json'))
    
    if not json_files:
        print("No JSON files found in /app/data/")
        return
    
    print(f"Processing {len(json_files)} files...")
    
    # CRITICAL BUG: Pool is created but never properly closed/joined
    # This causes resource leaks - semaphore objects are not cleaned up
    pool = Pool(4)
    results = pool.map(process_file, json_files)
    # Missing: pool.close() and pool.join()
    
    # Write results to output file
    output_file = output_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete. Results written to {output_file}")
    print(f"Processed {len(results)} files successfully")


if __name__ == '__main__':
    main()