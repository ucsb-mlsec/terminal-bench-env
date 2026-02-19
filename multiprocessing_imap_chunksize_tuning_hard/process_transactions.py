#!/usr/bin/env python3

import csv
import hashlib
import multiprocessing
import time
import sys


def verify_transaction(record):
    """
    Verify a transaction record by computing SHA256 hash of concatenated fields.
    
    Args:
        record: Dictionary containing transaction data
        
    Returns:
        bool: True if hash verification succeeds
    """
    # Concatenate the fields for hash verification
    data_string = f"{record['id']}{record['amount']}{record['timestamp']}{record['account']}"
    
    # Compute SHA256 hash
    computed_hash = hashlib.sha256(data_string.encode()).hexdigest()
    
    # Verify against stored hash
    return computed_hash == record['hash']


def process_file(filepath, chunk_size=None):
    """
    Process a CSV transaction file using multiprocessing.
    
    Args:
        filepath: Path to the CSV file to process
        chunk_size: Optional chunk size for pool.map()
        
    Returns:
        int: Number of records processed
    """
    # Read all records from CSV
    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    # Create a multiprocessing pool with default settings
    with multiprocessing.Pool() as pool:
        # Process records in parallel
        if chunk_size is not None:
            results = pool.map(verify_transaction, records, chunksize=chunk_size)
        else:
            results = pool.map(verify_transaction, records)
    
    return len(records)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_transactions.py <filepath> [chunk_size]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    chunk_size = None
    
    if len(sys.argv) >= 3:
        try:
            chunk_size = int(sys.argv[2])
        except ValueError:
            print("Error: chunk_size must be an integer")
            sys.exit(1)
    
    # Start timing
    start_time = time.time()
    
    # Process the file
    num_records = process_file(filepath, chunk_size)
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processed {num_records} records in {processing_time:.4f} seconds")
    if chunk_size is not None:
        print(f"Chunk size: {chunk_size}")