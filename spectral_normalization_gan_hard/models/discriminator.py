import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers - only some have spectral_norm applied
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
        # Fully connected layers - none have spectral_norm applied
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
        # Activation function
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        # Pass through convolutional layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x