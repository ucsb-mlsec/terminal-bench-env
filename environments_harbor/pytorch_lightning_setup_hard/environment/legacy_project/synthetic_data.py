#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    A synthetic dataset for regression tasks.
    Generates random input features and computes targets as a linear combination plus noise.
    """
    
    def __init__(self, num_samples, input_size, seed=None):
        """
        Initialize the synthetic dataset.
        
        Args:
            num_samples (int): Number of samples to generate
            input_size (int): Dimensionality of input features
            seed (int, optional): Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.input_size = input_size
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate random input features from normal distribution
        self.X = torch.randn(num_samples, input_size)
        
        # Create fixed weights for computing targets
        weights = torch.randn(input_size, 1)
        
        # Compute targets as linear combination plus noise
        noise = torch.randn(num_samples, 1) * 0.1
        self.Y = torch.matmul(self.X, weights) + noise
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (input_features, target_value)
        """
        return self.X[idx], self.Y[idx]