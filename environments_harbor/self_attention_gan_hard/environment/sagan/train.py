#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
from model import SelfAttention, Generator

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a SelfAttention module instance
attn = SelfAttention(in_channels=256)

print('Testing with 64x64 images (original resolution)...')

# Create test tensor for 64x64
x_64 = torch.randn(4, 256, 64, 64)

# Try-except block for 64x64
try:
    output_64 = attn(x_64)
    print('64x64 test PASSED')
except RuntimeError as e:
    print(f'ERROR: {str(e)}')

print('Testing with 128x128 images (new resolution)...')

# Create test tensor for 128x128
x_128 = torch.randn(4, 256, 128, 128)

# Try-except block for 128x128
try:
    output_128 = attn(x_128)
    print('128x128 test PASSED')
except RuntimeError as e:
    print(f'ERROR: {str(e)}')
    print('This is the dimension mismatch bug!')

print('\nThe error occurs in the reshape operation within the self-attention module.')
print('The dimensions are being calculated incorrectly for different image resolutions.')