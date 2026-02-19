#!/usr/bin/env python3

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    A replay buffer for storing samples from previous batches.
    This infrastructure exists but needs to be properly utilized in training.
    """
    
    def __init__(self, capacity=1000):
        """
        Initialize the replay buffer with a maximum capacity.
        
        Args:
            capacity (int): Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer = []
    
    def add(self, x, y):
        """
        Add a sample (x, y) to the buffer.
        If buffer exceeds capacity, remove the oldest sample.
        
        Args:
            x: Input feature value
            y: Target value
        """
        self.buffer.append((x, y))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        """
        Randomly sample batch_size items from the buffer.
        
        Args:
            batch_size (int): Number of samples to retrieve
            
        Returns:
            tuple: (x_array, y_array) - two numpy arrays of sampled data
        """
        batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, batch_size)
        
        x_array = np.array([s[0] for s in samples])
        y_array = np.array([s[1] for s in samples])
        
        return x_array, y_array
    
    def __len__(self):
        """
        Return the current number of samples in the buffer.
        
        Returns:
            int: Current buffer size
        """
        return len(self.buffer)
    
    def is_empty(self):
        """
        Check if the buffer is empty.
        
        Returns:
            bool: True if buffer is empty, False otherwise
        """
        return len(self.buffer) == 0