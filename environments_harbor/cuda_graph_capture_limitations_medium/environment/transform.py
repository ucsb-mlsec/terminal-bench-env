#!/usr/bin/env python3

import json
import numpy as np
import os
import sys

# Global cache for optimization
optimization_cache = {
    'initialized': False,
    'array_shape': None,
    'flat_indices': None,
    'operation_plan': None
}

def load_config():
    """Load configuration file."""
    config_path = '/workspace/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {'optimization_enabled': False}

def load_input_file(filepath):
    """Load 2D array from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return np.array(data['array'])

def transform_with_optimization(arr):
    """Transform array using cached optimization plan."""
    global optimization_cache
    
    # Initialize cache on first run
    if not optimization_cache['initialized']:
        optimization_cache['array_shape'] = arr.shape
        optimization_cache['flat_indices'] = np.arange(arr.size)
        optimization_cache['operation_plan'] = {
            'total_elements': arr.size,
            'shape': arr.shape
        }
        optimization_cache['initialized'] = True
        print(f"Optimization cache initialized for shape: {arr.shape}")
    
    # BUG: This assumes all arrays have the same shape as the first one
    # It will crash if array dimensions differ
    cached_shape = optimization_cache['array_shape']
    if arr.shape != cached_shape:
        # This will cause an error when trying to use cached indices
        print(f"Warning: Array shape {arr.shape} differs from cached {cached_shape}")
    
    # Use cached flat indices - this breaks with different sizes
    flat_arr = arr.flatten()
    indices = optimization_cache['flat_indices']
    
    # Apply operations using cached plan (this fails if size mismatches)
    result = np.zeros(optimization_cache['operation_plan']['total_elements'])
    result[indices] = flat_arr[indices]  # This line crashes if sizes differ
    
    # Apply transformations
    result = result * 2
    result = result + 10
    result = np.sqrt(result)
    
    # Reshape back to cached shape (wrong if input shape differs)
    result = result.reshape(cached_shape)
    
    return result

def transform_without_optimization(arr):
    """Transform array without optimization."""
    # Multiply by 2
    result = arr * 2
    # Add 10
    result = result + 10
    # Take square root
    result = np.sqrt(result)
    return result

def process_file(filepath, optimization_enabled):
    """Process a single input file."""
    print(f"Processing: {filepath}")
    arr = load_input_file(filepath)
    print(f"  Input shape: {arr.shape}")
    
    if optimization_enabled:
        result = transform_with_optimization(arr)
    else:
        result = transform_without_optimization(arr)
    
    print(f"  Output shape: {result.shape}")
    return result

def main():
    """Main processing function."""
    config = load_config()
    optimization_enabled = config.get('optimization_enabled', False)
    
    print(f"Optimization mode: {'ENABLED' if optimization_enabled else 'DISABLED'}")
    
    input_dir = '/workspace/inputs/'
    
    # Get all JSON files in input directory
    input_files = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.json')
    ])
    
    results = {}
    
    for filepath in input_files:
        try:
            result = process_file(filepath, optimization_enabled)
            results[os.path.basename(filepath)] = {
                'status': 'success',
                'shape': result.shape,
                'sample_values': result.flatten()[:5].tolist()
            }
        except Exception as e:
            print(f"ERROR processing {filepath}: {str(e)}")
            results[os.path.basename(filepath)] = {
                'status': 'failed',
                'error': str(e)
            }
            if optimization_enabled:
                print("Optimization mode failed due to dimension mismatch!")
            raise
    
    # Save results summary
    output_dir = '/workspace/outputs/'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nProcessing complete!")
    return results

if __name__ == '__main__':
    main()