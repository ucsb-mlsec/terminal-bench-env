import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(ImageClassifier, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size: 224 -> 112 -> 56 -> 28
        # So: 128 * 28 * 28 = 100352
        self.flatten = nn.Flatten()
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x
```

```python
#!/usr/bin/env python3

import torch
import sys
sys.path.append('/workspace')
from model_def import ImageClassifier

# Instantiate the model
model = ImageClassifier(num_classes=1000)

# Set model to evaluation mode
model.eval()

# Create dummy input with the specified shape [1, 3, 224, 224]
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(
    model,                          # Model to export
    dummy_input,                    # Example input tensor
    '/workspace/classifier.onnx',   # Output file path
    export_params=True,             # Store trained parameter weights
    opset_version=11,               # ONNX opset version
    do_constant_folding=True,       # Optimize constant folding
    input_names=['input'],          # Input tensor name
    output_names=['output'],        # Output tensor name
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model successfully exported to /workspace/classifier.onnx")