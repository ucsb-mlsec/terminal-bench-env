#!/usr/bin/env python3

import joblib
from joblib import Parallel, delayed
import numpy as np
import time
import sys
import os

def process_batch(batch_id):
    """
    Process a single batch of numerical data.
    Performs various computations on randomly generated data.
    """
    # Generate random numerical data
    data = np.random.randn(1000)
    
    # Perform computations
    mean_val = np.mean(data)
    std_val = np.std(data)
    sum_val = np.sum(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Simulate some processing time
    time.sleep(0.01)
    
    # Create nested operations that might trigger resource issues
    intermediate = np.array([mean_val, std_val, sum_val, min_val, max_val])
    result_data = np.cumsum(intermediate)
    
    return {
        'batch_id': batch_id,
        'mean': mean_val,
        'std': std_val,
        'sum': sum_val,
        'min': min_val,
        'max': max_val,
        'result': result_data.tolist()
    }

if __name__ == '__main__':
    print('Starting data processing pipeline...')
    print(f'Processing 100 batches using {os.cpu_count()} CPUs')
    
    start_time = time.time()
    
    # Use joblib with loky backend and all available CPUs
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_batch)(i) for i in range(100)
    )
    
    elapsed_time = time.time() - start_time
    
    print(f'Pipeline completed successfully!')
    print(f'Processed {len(results)} batches in {elapsed_time:.2f} seconds')
    print(f'First batch result: {results[0]["batch_id"]}, mean: {results[0]["mean"]:.4f}')
    print(f'Last batch result: {results[-1]["batch_id"]}, mean: {results[-1]["mean"]:.4f}')