#!/usr/bin/env python3
"""
Production data processing application entry point.
This script orchestrates the multiprocessing pipeline for large dataset processing.
"""

import multiprocessing
import time
from typing import List, Any

# Import global configuration and database connections
import config
from processor import process_chunk


def load_data_chunks() -> List[List[int]]:
    """
    Simulate loading large dataset chunks for processing.
    In production, this would load from database or file system.
    """
    print(f"Loading data chunks using DB connection: {config.DB_CONNECTION}")
    # Simulate 100 data items split into chunks of 10
    return [[i * 10 + j for j in range(10)] for i in range(10)]


def main():
    """
    Main processing pipeline that distributes work across multiple processes.
    Uses global config and database connections initialized at module level.
    """
    print("Starting production data processing pipeline...")
    print(f"Using configuration: {config.APP_NAME}")
    
    # Reference global state before creating worker processes
    # This pattern works differently with fork vs spawn
    db_conn = config.DB_CONNECTION
    print(f"Main process using DB connection: {db_conn}")
    
    # Load data chunks to process
    data_chunks = load_data_chunks()
    print(f"Loaded {len(data_chunks)} chunks for processing")
    
    # Create multiprocessing pool with 4 workers
    # Note: Using default start method (fork on Linux, spawn on macOS)
    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        print("Worker pool created, distributing work...")
        
        # Map processing function across all chunks
        # Workers will need access to config.DB_CONNECTION and other globals
        results = pool.map(process_chunk, data_chunks)
        
        print(f"Processing complete in {time.time() - start_time:.2f} seconds")
    
    # Collect and display results
    total_processed = sum(results)
    print(f"Successfully processed {total_processed} total items")
    print("Pipeline execution completed successfully")


if __name__ == '__main__':
    # Production entry point
    main()