import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```python
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Import the ImageClassifier from classifier.py
import sys
sys.path.insert(0, '/workspace')
from classifier import ImageClassifier

# Set seed for reproducibility
torch.manual_seed(42)

# Create model instance
model = ImageClassifier()

# Load the trained weights
model.load_state_dict(torch.load('/workspace/weights.pth', map_location='cpu'))

# Set model to evaluation mode
model.eval()

# Load test sample
test_sample = torch.load('/workspace/test_sample.pt', map_location='cpu')

# Validate the model produces expected output before export
with torch.no_grad():
    output = model(test_sample)
    predicted_class = output.argmax(dim=1).item()
    print(f"Pre-export validation - Predicted class: {predicted_class}")

# Export model using TorchScript for deployment without class definition
# Use torch.jit.trace for better compatibility with edge devices
model_traced = torch.jit.trace(model, test_sample)

# Save the traced model
production_model_path = '/workspace/production_model.pt'
torch.jit.save(model_traced, production_model_path)

print(f"Model exported to {production_model_path}")

# Verify the exported model works correctly
loaded_model = torch.jit.load(production_model_path, map_location='cpu')
loaded_model.eval()

# Run inference on test sample
with torch.no_grad():
    output = loaded_model(test_sample)
    predicted_class = output.argmax(dim=1).item()
    print(f"Post-export validation - Predicted class: {predicted_class}")

# Get file size in KB
file_size_bytes = os.path.getsize(production_model_path)
file_size_kb = file_size_bytes / 1024.0

# Write result to result.txt
result_path = '/workspace/result.txt'
with open(result_path, 'w') as f:
    f.write(f"artifact_size_kb={file_size_kb:.1f}\n")
    f.write(f"predicted_class={predicted_class}\n")

print(f"\nDeployment complete!")
print(f"Artifact size: {file_size_kb:.1f} KB")
print(f"Predicted class: {predicted_class}")
print(f"Result saved to {result_path}")