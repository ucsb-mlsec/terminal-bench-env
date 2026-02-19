#!/usr/bin/env python3
import numpy as np
import math

class PositionalEncoding:
    """
    Positional Encoding class for transformer models.
    Generates sinusoidal positional encodings.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the PositionalEncoding.
        
        Args:
            d_model: The embedding dimension
            max_len: Maximum sequence length to pre-compute
        """
        self.d_model = d_model
        self.max_len = max_len
    
    def get_encoding(self, seq_len):
        """
        Generate positional encoding for a given sequence length.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Positional encoding matrix of shape (seq_len, d_model)
        """
        # Create position indices
        position = np.arange(0, seq_len).reshape(-1, 1)
        
        # Create dimension indices - BUG: using i instead of i//2
        div_term = np.arange(0, self.d_model, 2)
        
        # Calculate frequencies - BUG: using 1000 instead of 10000
        div_term = np.exp(div_term * -(math.log(1000.0) / self.d_model))
        
        # Initialize encoding matrix
        pe = np.zeros((seq_len, self.d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cosine to odd indices - BUG: incorrect slicing if d_model is odd
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe