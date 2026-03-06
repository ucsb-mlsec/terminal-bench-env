#!/usr/bin/env python3

import torch
import os
from baseline_model import get_model

# Create the model instance
model = get_model()

# Save the model's state_dict
model_path = 'baseline_fp32.pth'
torch.save(model.state_dict(), model_path)

# Get file size and print confirmation
file_size_bytes = os.path.getsize(model_path)
file_size_kb = file_size_bytes / 1024

print(f'Baseline FP32 model saved to {model_path}')
print(f'File size: {file_size_kb:.1f} KB')