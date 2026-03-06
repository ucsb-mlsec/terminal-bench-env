import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """
    Simple feedforward neural network for regression tasks.
    
    Architecture:
    - Input layer to hidden layer (fc1)
    - ReLU activation
    - Hidden layer to hidden layer (fc2)
    - ReLU activation
    - Hidden layer to output layer (fc3)
    """
    
    def __init__(self, input_size=1, hidden_size=64):
        """
        Initialize the model.
        
        Args:
            input_size (int): Number of input features (default: 1)
            hidden_size (int): Number of hidden units (default: 64)
        """
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x