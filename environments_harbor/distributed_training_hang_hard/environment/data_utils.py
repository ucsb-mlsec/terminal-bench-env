#!/usr/bin/env python3
"""
Data loading utilities for distributed training.
"""

import torch
import torch.utils.data
import numpy as np
import torch.distributed as dist


class SimpleDataset(torch.utils.data.Dataset):
    """
    A simple dataset that generates random data.
    """
    
    def __init__(self, size, input_dim=10):
        """
        Initialize the dataset.
        
        Args:
            size: Number of samples in the dataset
            input_dim: Dimensionality of input features
        """
        self.size = size
        self.input_dim = input_dim
    
    def __getitem__(self, idx):
        """
        Generate a random sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (data tensor, label)
        """
        data = torch.randn(self.input_dim)
        label = torch.randint(0, 2, (1,)).item()
        return data, label
    
    def __len__(self):
        """Return the size of the dataset."""
        return self.size


def create_dataset(dataset_size, input_dim=10):
    """
    Create a synthetic dataset.
    
    Args:
        dataset_size: Number of samples
        input_dim: Dimensionality of input features
        
    Returns:
        SimpleDataset instance
    """
    return SimpleDataset(dataset_size, input_dim)


def get_data_loader(batch_size, num_workers, world_size, rank, dataset_size=10000):
    """
    Create a distributed data loader.
    
    Args:
        batch_size: Batch size per process
        num_workers: Number of data loading workers
        world_size: Total number of processes
        rank: Rank of current process
        dataset_size: Total dataset size
        
    Returns:
        DataLoader configured for distributed training
    """
    dataset = SimpleDataset(dataset_size)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False
    )
    
    return loader