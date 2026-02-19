# Baseline Generator Architecture (DO NOT MODIFY THIS FILE)
# Input: latent_dim=128
# Output: 64x64 RGB images
# 
# Architecture:
# - Linear: 128 -> 8192 (reshaped to 512x4x4)
# - ConvTranspose2d: 512 -> 256, kernel=4, stride=2, padding=1 (output: 256x8x8)
# - ConvTranspose2d: 256 -> 128, kernel=4, stride=2, padding=1 (output: 128x16x16)
# - ConvTranspose2d: 128 -> 64, kernel=4, stride=2, padding=1 (output: 64x32x32)
# - ConvTranspose2d: 64 -> 3, kernel=4, stride=2, padding=1 (output: 3x64x64)
# - Tanh activation on final output
# - BatchNorm and ReLU between transpose conv layers
#
# Total Parameters: 2,769,731

import torch
import torch.nn as nn


class BaselineGenerator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, img_size=64):
        super(BaselineGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Linear layer: 128 -> 8192 (reshaped to 512x4x4)
        self.linear = nn.Linear(latent_dim, 8192)
        
        # ConvTranspose2d: 512 -> 256, kernel=4, stride=2, padding=1 (output: 256x8x8)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        
        # ConvTranspose2d: 256 -> 128, kernel=4, stride=2, padding=1 (output: 128x16x16)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        
        # ConvTranspose2d: 128 -> 64, kernel=4, stride=2, padding=1 (output: 64x32x32)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        # ConvTranspose2d: 64 -> 3, kernel=4, stride=2, padding=1 (output: 3x64x64)
        self.conv4 = nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, z):
        # z shape: (batch_size, 128)
        x = self.linear(z)
        
        # Reshape to (batch_size, 512, 4, 4)
        x = x.view(-1, 512, 4, 4)
        
        # ConvTranspose2d: 512 -> 256 (output: 256x8x8)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # ConvTranspose2d: 256 -> 128 (output: 128x16x16)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # ConvTranspose2d: 128 -> 64 (output: 64x32x32)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # ConvTranspose2d: 64 -> 3 (output: 3x64x64)
        x = self.conv4(x)
        x = self.tanh(x)
        
        return x