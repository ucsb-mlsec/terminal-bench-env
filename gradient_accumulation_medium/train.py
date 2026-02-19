#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Model Definition
class SimpleNet(nn.Module):
    """3-layer neural network for binary classification"""
    def __init__(self, input_size=20, hidden1=64, hidden2=32):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def create_synthetic_dataset(num_samples=1000, num_features=20):
    """Create synthetic dataset for binary classification"""
    # Generate random features
    X = torch.randn(num_samples, num_features)
    # Generate random binary labels
    y = torch.randint(0, 2, (num_samples, 1)).float()
    return X, y

def train_model(model, dataloader, criterion, optimizer, epochs=1):
    """Standard training loop"""
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (weight update after every batch)
            optimizer.step()
            
            # Zero gradients for next iteration
            optimizer.zero_grad()
            
            # Track loss
            epoch_loss += loss.item()
            batch_count += 1
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # Print epoch summary
        avg_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    # Training configuration
    batch_size = 64
    learning_rate = 0.001
    epochs = 1
    num_samples = 1000
    num_features = 20
    
    print("Initializing training...")
    print(f"Configuration: batch_size={batch_size}, learning_rate={learning_rate}, epochs={epochs}")
    
    # Create synthetic dataset
    print(f"\nCreating synthetic dataset: {num_samples} samples, {num_features} features")
    X, y = create_synthetic_dataset(num_samples=num_samples, num_features=num_features)
    dataset = TensorDataset(X, y)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with batch_size={batch_size}, total batches per epoch: {len(dataloader)}")
    
    # Instantiate model
    model = SimpleNet(input_size=num_features)
    print(f"\nModel architecture:\n{model}")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("\nStarting training...\n")
    train_model(model, dataloader, criterion, optimizer, epochs=epochs)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)