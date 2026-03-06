#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os


def load_data(file_path):
    """
    Load data from CSV file and return x and y arrays.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        x: numpy array of x values, reshaped to (-1, 1)
        y: numpy array of y values
    """
    df = pd.read_csv(file_path)
    x = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    return x, y


def evaluate_model(model, test_file_path):
    """
    Evaluate model on test data and return MSE.
    
    Args:
        model: PyTorch model
        test_file_path: Path to test CSV file
        
    Returns:
        mse: Mean squared error on test set
    """
    # Load test data
    x, y = load_data(test_file_path)
    
    # Convert to torch tensors
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)
    
    # Set model to evaluation mode and get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor)
        predictions = predictions.numpy().flatten()
    
    # Compute MSE
    mse = mean_squared_error(y, predictions)
    
    return mse


def evaluate_all_batches(model, data_dir):
    """
    Evaluate model on all three batch test sets.
    
    Args:
        model: PyTorch model
        data_dir: Directory containing test data files
        
    Returns:
        dict: Dictionary with MSE for each batch and average
    """
    batch1_test = os.path.join(data_dir, 'batch1_test.csv')
    batch2_test = os.path.join(data_dir, 'batch2_test.csv')
    batch3_test = os.path.join(data_dir, 'batch3_test.csv')
    
    batch1_mse = evaluate_model(model, batch1_test)
    batch2_mse = evaluate_model(model, batch2_test)
    batch3_mse = evaluate_model(model, batch3_test)
    
    average_mse = (batch1_mse + batch2_mse + batch3_mse) / 3.0
    
    results = {
        'batch1_mse': float(batch1_mse),
        'batch2_mse': float(batch2_mse),
        'batch3_mse': float(batch3_mse),
        'average_mse': float(average_mse)
    }
    
    return results