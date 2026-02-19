#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure wandb for offline mode
wandb.init(mode='offline', project='ml-training', name='experiment-demo')

# Create synthetic training data
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randint(0, 2, size=(100,))

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y).long()

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, optimizer, and loss function
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
for epoch in range(10):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log metrics - BROKEN: wandb.log() is commented out
    # wandb.log({"loss": loss.item()})
    
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

print("Training completed!")

# MISSING: wandb.finish() call is not present