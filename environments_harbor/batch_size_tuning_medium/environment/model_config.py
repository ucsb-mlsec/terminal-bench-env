#!/usr/bin/env python3

import torch
import torch.nn as nn

# Current batch size configuration (set too high, needs adjustment)
BATCH_SIZE = 128

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for image classification on 224x224 RGB images.
    
    Architecture:
    - Conv layers with increasing channels: 64 -> 128 -> 256 -> 512
    - Batch normalization after each conv layer
    - ReLU activations
    - Max pooling for spatial downsampling
    - Fully connected layers: 512*7*7 -> 2048 -> 1024 -> 1000
    """
    
    def __init__(self, num_classes=1000):
        super(CNNModel, self).__init__()
        
        # Convolutional Block 1: 3 -> 64 channels
        # Input: (B, 3, 224, 224) -> Output: (B, 64, 112, 112)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2: 64 -> 128 channels
        # Input: (B, 64, 112, 112) -> Output: (B, 128, 56, 56)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3: 128 -> 256 channels
        # Input: (B, 128, 56, 56) -> Output: (B, 256, 28, 28)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4: 256 -> 512 channels
        # Input: (B, 256, 28, 28) -> Output: (B, 512, 14, 14)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 5: 512 -> 512 channels
        # Input: (B, 512, 14, 14) -> Output: (B, 512, 7, 7)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # After pooling, spatial size is 7x7, so flatten to 512*7*7 = 25088
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Conv Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_num_parameters(self):
        """
        Calculate and return the total number of trainable parameters.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_info():
    """
    Get detailed information about the model architecture.
    
    Returns:
        dict: Dictionary containing model information
    """
    model = CNNModel()
    num_params = model.get_num_parameters()
    
    # Calculate parameter memory (FP32 = 4 bytes per parameter)
    param_memory_mb = (num_params * 4) / (1024 * 1024)
    
    info = {
        'model_name': 'CNNModel',
        'input_size': (3, 224, 224),
        'num_classes': 1000,
        'total_parameters': num_params,
        'parameter_memory_mb': param_memory_mb,
        'current_batch_size': BATCH_SIZE,
        'architecture': {
            'conv_layers': 5,
            'fc_layers': 3,
            'channels': [64, 128, 256, 512, 512]
        }
    }
    
    return info


if __name__ == '__main__':
    # Create model and display information
    model = CNNModel()
    info = get_model_info()
    
    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"Model Name: {info['model_name']}")
    print(f"Input Size: {info['input_size']}")
    print(f"Number of Classes: {info['num_classes']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Parameter Memory: {info['parameter_memory_mb']:.2f} MB")
    print(f"Current Batch Size: {info['current_batch_size']}")
    print(f"\nArchitecture: {info['architecture']}")
    print("=" * 60)