import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        
        x = self.conv3(x)
        x = self.sigmoid(x)
        
        return x