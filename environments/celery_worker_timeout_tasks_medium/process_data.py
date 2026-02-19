#!/usr/bin/env python3

import time
import os

def process_file(filename):
    """
    Simulates data processing with time delays based on file size.
    
    Args:
        filename: Name of the file to process
        
    Returns:
        Success message with processing details
    """
    basename = os.path.basename(filename).lower()
    
    if 'small' in basename:
        processing_time = 5
    elif 'medium' in basename:
        processing_time = 45
    elif 'large' in basename:
        processing_time = 90
    else:
        processing_time = 10
    
    print(f"Processing {filename}... (estimated {processing_time}s)")
    time.sleep(processing_time)
    
    return f"Successfully processed {filename} in {processing_time} seconds"

if __name__ == "__main__":
    # Test processing with different file sizes
    test_files = ['small.dat', 'medium.dat', 'large.dat']
    for test_file in test_files:
        result = process_file(test_file)
        print(result)