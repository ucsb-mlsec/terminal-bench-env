import torch.nn as nn
import torch

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Create a convolutional block with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def init_weights(module):
    """Initialize weights for Conv2d and Linear layers."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def residual_block(channels):
    """Create a residual block."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, 1, 1),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 3, 1, 1),
        nn.BatchNorm2d(channels)
    )

def create_encoder_block(in_channels, out_channels, normalize=True):
    """Create an encoder block for downsampling."""
    layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def create_decoder_block(in_channels, out_channels, dropout=False):
    """Create a decoder block for upsampling."""
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)