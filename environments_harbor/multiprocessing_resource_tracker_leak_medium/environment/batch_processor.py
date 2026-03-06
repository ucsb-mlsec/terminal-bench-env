#!/usr/bin/env python3

import os
import csv
import json
import multiprocessing
from multiprocessing import Pool, Manager, Queue

def process_file(filepath):
    """Process a single CSV file and calculate statistics."""
    try:
        values = []
        record_count = 0
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                values.append(float(row['value']))
                record_count += 1
        
        if values:
            mean_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
        else:
            mean_val = min_val = max_val = 0
        
        return {
            'filename': os.path.basename(filepath),
            'mean': mean_val,
            'min': min_val,
            'max': max_val,
            'record_count': record_count
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    data_dir = '/workspace/data/'
    output_file = '/workspace/results.json'
    
    try:
        # Get all CSV files
        csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found")
            return
        
        # Create multiprocessing resources WITHOUT proper cleanup
        pool = Pool(4)  # BUG: Not using context manager
        manager = Manager()  # BUG: Never terminated
        queue = manager.Queue()  # BUG: Created but not cleaned up
        shared_list = manager.list()  # BUG: Created but not cleaned up
        
        # Process files in parallel
        results = pool.map(process_file, csv_files)
        
        # BUG: Missing pool.close() and pool.join()
        
        # Calculate totals
        processed_files = 0
        total_records = 0
        
        for result in results:
            if result:
                processed_files += 1
                total_records += result['record_count']
                shared_list.append(result)  # Using shared resource unnecessarily
        
        # Write results
        output_data = {
            'processed_files': processed_files,
            'total_records': total_records,
            'status': 'success'
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Processing complete: {processed_files} files, {total_records} records")
        
    except Exception as e:
        print(f"Error in main: {e}")
        # BUG: Even in error case, resources are not cleaned up
        output_data = {
            'processed_files': 0,
            'total_records': 0,
            'status': 'error'
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

if __name__ == '__main__':
    main()