#!/usr/bin/env python3

import torch
import torch.onnx
from model import CustomResNet

print('Starting ONNX export...')

# Instantiate the model
model = CustomResNet()

# Set to evaluation mode
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX - with bugs
torch.onnx.export(
    model,
    dummy_input,
    '/workspace/model.onnx',
    opset_version=9,
    export_params=True
)

print('Export completed')