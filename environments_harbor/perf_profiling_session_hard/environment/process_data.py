#!/usr/bin/env python3

import numpy as np
import time
import sys

def load_data():
    """Load and initialize data"""
    print("Loading data...")
    data = np.random.rand(1000, 1000)
    print(f"Loaded data with shape {data.shape}")
    return data

def transform_data(data):
    """Transform data using matrix operations"""
    print("Transforming data...")
    result = np.sqrt(data) * 2.0 + np.sin(data)
    print("Transformation complete")
    return result

def calculate_stats(data):
    """Calculate statistical measures for each element"""
    print("Calculating statistics...")
    rows, cols = data.shape
    result = np.zeros_like(data)
    
    # Inefficient element-by-element processing using Python loops
    # instead of vectorized numpy operations
    for i in range(rows):
        for j in range(cols):
            value = data[i, j]
            # Perform redundant calculations
            temp = 0
            for k in range(100):
                temp += value * np.sin(value + k * 0.01)
                temp += value * np.cos(value + k * 0.01)
                temp = temp / 2.0
            result[i, j] = temp
    
    print("Statistics calculation complete")
    return result

def summarize_results(data):
    """Summarize the processed data"""
    print("Summarizing results...")
    summary = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    print(f"Summary: mean={summary['mean']:.4f}, std={summary['std']:.4f}")
    return summary

def main():
    """Main processing pipeline"""
    print("=" * 50)
    print("Data Processing Application")
    print("=" * 50)
    
    start_time = time.time()
    
    # Load data
    data = load_data()
    
    # Transform data
    transformed = transform_data(data)
    
    # Calculate statistics (bottleneck)
    stats_data = calculate_stats(transformed)
    
    # Summarize results
    summary = summarize_results(stats_data)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("=" * 50)
    print(f"Processing complete in {elapsed:.2f} seconds")
    print("=" * 50)

if __name__ == '__main__':
    main()