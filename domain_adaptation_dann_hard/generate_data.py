#!/usr/bin/env python3

import numpy as np
import pandas as pd

def generate_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_train = 2000
    n_test = 500
    shift = 2.0
    noise_level = 0.3
    label_flip_prob = 0.05
    
    # Generate source domain training data
    source_train_x = np.random.randn(n_train, 2)
    source_train_labels = (source_train_x[:, 0] + source_train_x[:, 1] > 0).astype(int)
    
    # Add label noise (flip some labels)
    flip_mask = np.random.rand(n_train) < label_flip_prob
    source_train_labels[flip_mask] = 1 - source_train_labels[flip_mask]
    
    # Generate source domain test data
    source_test_x = np.random.randn(n_test, 2)
    source_test_labels = (source_test_x[:, 0] + source_test_x[:, 1] > 0).astype(int)
    
    # Add label noise
    flip_mask = np.random.rand(n_test) < label_flip_prob
    source_test_labels[flip_mask] = 1 - source_test_labels[flip_mask]
    
    # Generate target domain training data
    # Generate base features, apply decision boundary, then shift
    target_train_x_base = np.random.randn(n_train, 2)
    target_train_labels = (target_train_x_base[:, 0] + target_train_x_base[:, 1] > 0).astype(int)
    
    # Add label noise
    flip_mask = np.random.rand(n_train) < label_flip_prob
    target_train_labels[flip_mask] = 1 - target_train_labels[flip_mask]
    
    # Apply shift to features
    target_train_x = target_train_x_base + shift
    
    # Generate target domain test data
    target_test_x_base = np.random.randn(n_test, 2)
    target_test_labels = (target_test_x_base[:, 0] + target_test_x_base[:, 1] > 0).astype(int)
    
    # Add label noise
    flip_mask = np.random.rand(n_test) < label_flip_prob
    target_test_labels[flip_mask] = 1 - target_test_labels[flip_mask]
    
    # Apply shift to features
    target_test_x = target_test_x_base + shift
    
    # Create DataFrames
    source_train_df = pd.DataFrame({
        'x1': source_train_x[:, 0],
        'x2': source_train_x[:, 1],
        'label': source_train_labels,
        'domain': np.zeros(n_train, dtype=int)
    })
    
    source_test_df = pd.DataFrame({
        'x1': source_test_x[:, 0],
        'x2': source_test_x[:, 1],
        'label': source_test_labels,
        'domain': np.zeros(n_test, dtype=int)
    })
    
    target_train_df = pd.DataFrame({
        'x1': target_train_x[:, 0],
        'x2': target_train_x[:, 1],
        'label': target_train_labels,
        'domain': np.ones(n_train, dtype=int)
    })
    
    target_test_df = pd.DataFrame({
        'x1': target_test_x[:, 0],
        'x2': target_test_x[:, 1],
        'label': target_test_labels,
        'domain': np.ones(n_test, dtype=int)
    })
    
    # Save to CSV files
    source_train_df.to_csv('/workspace/source_train.csv', index=False)
    source_test_df.to_csv('/workspace/source_test.csv', index=False)
    target_train_df.to_csv('/workspace/target_train.csv', index=False)
    target_test_df.to_csv('/workspace/target_test.csv', index=False)
    
    print("Data generation complete!")
    print(f"Source train: {len(source_train_df)} samples")
    print(f"Source test: {len(source_test_df)} samples")
    print(f"Target train: {len(target_train_df)} samples")
    print(f"Target test: {len(target_test_df)} samples")

if __name__ == '__main__':
    generate_data()