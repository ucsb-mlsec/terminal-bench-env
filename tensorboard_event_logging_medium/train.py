#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
print("Generating synthetic data...")
X = np.random.randn(1000, 10).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.float32)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).unsqueeze(1)
X_val_tensor = torch.from_numpy(X_val)
y_val_tensor = torch.from_numpy(y_val).unsqueeze(1)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, loss function, and optimizer
model = SimpleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nStarting training...")
epochs = 5

for epoch in range(epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        
        # Calculate accuracy
        train_predictions = (outputs > 0.5).float()
        train_accuracy = (train_predictions == y_train_tensor).float().mean().item()
        
        val_predictions = (val_outputs > 0.5).float()
        val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
        
        # Calculate precision
        train_precision = precision_score(
            y_train_tensor.numpy(), 
            train_predictions.numpy(), 
            zero_division=0
        )
        val_precision = precision_score(
            y_val_tensor.numpy(), 
            val_predictions.numpy(), 
            zero_division=0
        )
    
    # Print metrics
    print(f"Epoch {epoch+1}/{epochs} - "
          f"Loss: {loss.item():.4f}, "
          f"Accuracy: {train_accuracy:.4f}, "
          f"Val_Loss: {val_loss.item():.4f}, "
          f"Val_Accuracy: {val_accuracy:.4f}, "
          f"Train_Precision: {train_precision:.4f}, "
          f"Val_Precision: {val_precision:.4f}")

print("\nTraining completed!")