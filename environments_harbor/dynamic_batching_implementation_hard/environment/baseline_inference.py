#!/usr/bin/env python3

import numpy as np
import json
import time
from model import Predictor

def main():
    # Read the queue file to get list of tensor filenames
    with open('/workspace/queue.txt', 'r') as f:
        tensor_files = [line.strip() for line in f.readlines()]
    
    # Initialize the predictor
    predictor = Predictor()
    
    # Dictionary to store results
    results = {}
    
    # Start timing
    start_time = time.time()
    
    # Process each tensor sequentially
    for tensor_file in tensor_files:
        # Load the tensor
        tensor_path = f'/workspace/tensors/{tensor_file}'
        tensor = np.load(tensor_path)
        
        # Run prediction on single tensor
        prediction = predictor.predict(tensor)
        
        # Store result rounded to 1 decimal place
        results[tensor_file] = round(float(prediction), 1)
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results to JSON
    with open('/workspace/predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save timing information
    with open('/workspace/baseline_time.txt', 'w') as f:
        f.write(f'Total processing time: {total_time:.2f} seconds\n')
        f.write(f'Average time per tensor: {total_time/len(tensor_files):.4f} seconds\n')
    
    # Print timing information
    print(f'Processed {len(tensor_files)} tensors sequentially')
    print(f'Total processing time: {total_time:.2f} seconds')
    print(f'Average time per tensor: {total_time/len(tensor_files):.4f} seconds')

if __name__ == '__main__':
    main()