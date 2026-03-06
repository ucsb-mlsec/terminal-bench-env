#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def generate_data(num_samples, input_dim, random_seed=42):
    """
    Generate synthetic binary classification data.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Number of input features
        random_seed: Random seed for reproducibility
        
    Returns:
        features: Tensor of shape (num_samples, input_dim)
        labels: Tensor of shape (num_samples,)
    """
    torch.manual_seed(random_seed)
    
    # Generate random features
    features = torch.randn(num_samples, input_dim, dtype=torch.float32)
    
    # Create a random weight vector for generating labels
    weights = torch.randn(input_dim, dtype=torch.float32)
    
    # Generate labels using a linear combination + sigmoid
    logits = torch.matmul(features, weights)
    probabilities = torch.sigmoid(logits)
    
    # Create binary labels (threshold at 0.5)
    labels = (probabilities > 0.5).float()
    
    return features, labels


def create_dataloader(features, labels, batch_size=32, shuffle=True):
    """
    Create a DataLoader from features and labels.
    
    Args:
        features: Input features tensor
        labels: Target labels tensor
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader object
    """
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader