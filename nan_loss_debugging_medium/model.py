import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """
    A simple feedforward neural network for binary classification.
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the binary classifier.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units in the first hidden layer
        """
        super(BinaryClassifier, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, 1)
        """
        # Pass through first layer
        x = self.fc1(x)
        x = self.relu1(x)
        
        # Pass through second layer
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Pass through output layer (no activation - returns logits)
        x = self.fc3(x)
        
        return x