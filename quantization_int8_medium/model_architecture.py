#!/usr/bin/env python3

import numpy as np


class MNISTModel:
    """Simple 3-layer neural network for MNIST digit classification."""
    
    def __init__(self):
        """Initialize the model structure."""
        self.weights = {}
        self.biases = {}
        
        # Define layer dimensions
        self.layer_dims = [
            (784, 128),  # Input to Hidden 1
            (128, 64),   # Hidden 1 to Hidden 2
            (64, 10)     # Hidden 2 to Output
        ]
        
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Softmax activation function."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def load_weights(self, weights_array):
        """
        Load weights from a flattened numpy array.
        
        Args:
            weights_array: 1D numpy array containing all weights and biases
        """
        idx = 0
        
        for i, (in_dim, out_dim) in enumerate(self.layer_dims):
            # Load weight matrix
            weight_size = in_dim * out_dim
            self.weights[f'layer{i+1}'] = weights_array[idx:idx+weight_size].reshape(in_dim, out_dim)
            idx += weight_size
            
            # Load bias vector
            bias_size = out_dim
            self.biases[f'layer{i+1}'] = weights_array[idx:idx+bias_size]
            idx += bias_size
    
    def forward(self, x):
        """
        Perform forward pass through the network.
        
        Args:
            x: Input array of shape (batch_size, 784) or (784,)
            
        Returns:
            Output probabilities of shape (batch_size, 10) or (10,)
        """
        # Handle single sample or batch
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        # Flatten if needed (28x28 -> 784)
        if x.shape[-1] != 784:
            x = x.reshape(x.shape[0], -1)
        
        # Layer 1: Input -> Hidden 1
        z1 = np.dot(x, self.weights['layer1']) + self.biases['layer1']
        a1 = self.relu(z1)
        
        # Layer 2: Hidden 1 -> Hidden 2
        z2 = np.dot(a1, self.weights['layer2']) + self.biases['layer2']
        a2 = self.relu(z2)
        
        # Layer 3: Hidden 2 -> Output
        z3 = np.dot(a2, self.weights['layer3']) + self.biases['layer3']
        output = self.softmax(z3)
        
        if single_sample:
            return output[0]
        return output
    
    def predict(self, x):
        """
        Predict class labels for input samples.
        
        Args:
            x: Input array of shape (batch_size, 784) or (784,)
            
        Returns:
            Predicted class labels
        """
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=-1)
    
    def get_total_parameters(self):
        """Calculate total number of parameters in the model."""
        total = 0
        for in_dim, out_dim in self.layer_dims:
            total += in_dim * out_dim + out_dim  # weights + biases
        return total