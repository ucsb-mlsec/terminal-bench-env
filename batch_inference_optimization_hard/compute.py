#!/usr/bin/env python3

import numpy as np
import time
import os

def main():
    # Start timing
    start_time = time.time()
    
    # Force single-threaded execution to slow things down
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    print("Loading data...")
    
    # Load weights inefficiently (with unnecessary copy)
    weights_temp = np.load('/data/matrices/weights.npy')
    weights = np.copy(weights_temp)  # Unnecessary copy
    del weights_temp
    
    # Load batch data inefficiently
    batch_data_temp = np.load('/data/matrices/batch_data.npy')
    batch_data = np.copy(batch_data_temp)  # Another unnecessary copy
    del batch_data_temp
    
    print(f"Loaded weights with shape: {weights.shape}")
    print(f"Loaded batch_data with shape: {batch_data.shape}")
    
    # Do some unnecessary preprocessing with loops (inefficient)
    print("Preprocessing data...")
    for i in range(min(100, batch_data.shape[0])):
        # Normalize each sample unnecessarily in a loop
        batch_data[i] = batch_data[i] / (np.linalg.norm(batch_data[i]) + 1e-10) * np.linalg.norm(batch_data[i])
    
    print("Performing matrix multiplication...")
    
    # Transpose weights (creating another copy)
    weights_transposed = weights.T
    
    # Perform matrix multiplication
    result = batch_data @ weights_transposed
    
    # Create unnecessary intermediate copy
    result_copy = np.copy(result)
    
    # Ensure output directory exists
    os.makedirs('/data/results', exist_ok=True)
    
    print("Saving results...")
    
    # Save result
    np.save('/data/results/output.npy', result_copy)
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution completed in {execution_time:.2f} seconds")
    print(f"Result shape: {result_copy.shape}")

if __name__ == "__main__":
    main()