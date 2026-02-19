#!/usr/bin/env python3
"""
Neural network model for binary classification on tabular data.
This model is designed to work with 10 input features and predict binary labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    """
    A feedforward neural network for binary classification.
    
    Architecture:
        - Input layer: 10 features
        - Hidden layer 1: 64 neurons with ReLU activation
        - Hidden layer 2: 32 neurons with ReLU activation
        - Output layer: 2 neurons (one for each class)
    
    Args:
        input_dim (int): Number of input features (default: 10)
    """
    
    def __init__(self, input_dim=10):
        super(BinaryClassifier, self).__init__()
        
        # Define the network layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Bug: Should be 1 for binary classification with BCEWithLogitsLoss
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        # First hidden layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation
        x = self.fc2(x)
        # Bug: Missing activation function here
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x