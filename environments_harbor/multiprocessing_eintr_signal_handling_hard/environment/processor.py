#!/usr/bin/env python3

import multiprocessing
import os
import sys
import time
import csv
import json
import signal

def worker_process(file_path, result_pipe):
    """
    Worker function that processes a CSV file.
    INTENTIONALLY BUGGY - no signal handling or EINTR retry logic.
    """
    print(f"Worker {os.getpid()} started processing {file_path}")
    
    # No signal handler registered - signals will interrupt operations
    
    row_count = 0
    sum_values = 0
    
    # Open file with no error handling for EINTR
    fd = os.open(file_path, os.O_RDONLY)
    file_obj = os.fdopen(fd, 'r')
    
    reader = csv.reader(file_obj)
    
    # Skip header
    next(reader)
    
    for row in reader:
        # Simulate long computation - blocking operation
        time.sleep(0.1)
        
        row_count += 1
        if len(row) > 1:
            try:
                sum_values += float(row[1])
            except:
                pass
        
        # Simulate some I/O operation that can be interrupted
        if row_count % 10 == 0:
            time.sleep(0.2)
    
    file_obj.close()
    
    result = {
        'file': os.path.basename(file_path),
        'rows': row_count,
        'sum': sum_values
    }
    
    print(f"Worker {os.getpid()} finished {file_path}: {row_count} rows")
    
    # Write result to pipe - no EINTR handling
    result_json = json.dumps(result)
    os.write(result_pipe, result_json.encode() + b'\n')


def process_files():
    """
    Main processing function - spawns workers and collects results.
    INTENTIONALLY BUGGY - no error handling for worker failures.
    """
    data_dir = '/opt/dataprocessor/data'
    output_dir = '/opt/dataprocessor/output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(data_dir, filename))
    
    print(f"Found {len(csv_files)} files to process")
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    # Create pipes for each worker - raw pipes without error handling
    workers = []
    pipes = []
    
    for file_path in csv_files:
        # Create pipe for this worker
        read_fd, write_fd = os.pipe()
        pipes.append(read_fd)
        
        # Spawn worker process
        p = multiprocessing.Process(target=worker_process, args=(file_path, write_fd))
        p.start()
        workers.append(p)
        
        # Close write end in parent
        os.close(write_fd)
    
    print(f"Spawned {len(workers)} worker processes")
    
    # Collect results from all workers - no EINTR handling on read
    results = []
    for read_fd in pipes:
        # Blocking read - will fail with EINTR if signal arrives
        data = b''
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            data += chunk
        
        os.close(read_fd)
        
        if data:
            result = json.loads(data.decode().strip())
            results.append(result)
    
    # Wait for all workers - no error checking
    for p in workers:
        p.join()
    
    # Aggregate results
    total_rows = sum(r['rows'] for r in results)
    total_sum = sum(r['sum'] for r in results)
    
    final_result = {
        'files_processed': len(results),
        'total_rows': total_rows,
        'total_sum': total_sum,
        'details': results
    }
    
    # Write output - no EINTR handling
    output_path = os.path.join(output_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print(f"Processing complete: {len(results)} files, {total_rows} total rows")
    print(f"Results written to {output_path}")


if __name__ == '__main__':
    print("Starting data processor...")
    print(f"Main process PID: {os.getpid()}")
    
    # No signal handling setup
    
    start_time = time.time()
    process_files()
    end_time = time.time()
    
    print(f"Total processing time: {end_time - start_time:.2f} seconds")