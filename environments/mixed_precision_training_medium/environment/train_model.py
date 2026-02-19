#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Simple neural network for demonstration
class SimpleNet(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def generate_synthetic_data(num_samples=1000, input_size=128, output_size=10):
    """Generate synthetic data for training"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    return X, y

def create_batches(X, y, batch_size=32):
    """Create batches from data"""
    num_samples = X.shape[0]
    indices = torch.randperm(num_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        batches.append((batch_X, batch_y))
    
    return batches

def train_model():
    """Main training function with BROKEN mixed precision implementation"""
    
    # Training configuration
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    input_size = 128
    hidden_size = 256
    output_size = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(num_samples=1000, 
                                                input_size=input_size, 
                                                output_size=output_size)
    
    # Initialize model, loss, and optimizer
    model = SimpleNet(input_size=input_size, 
                      hidden_size=hidden_size, 
                      output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    print("Starting training with mixed precision enabled...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Create batches for this epoch
        batches = create_batches(X_train, y_train, batch_size=batch_size)
        
        for batch_idx, (batch_X, batch_y) in enumerate(batches):
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # ERROR 1: Missing autocast context around forward pass
            # The forward pass should be wrapped in autocast for mixed precision
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # ERROR 2: Autocast incorrectly used around backward pass
            # autocast should NOT wrap the backward pass
            with autocast():
                # ERROR 3: scaler.scale() should be called on loss, not after backward
                loss.backward()
            
            # ERROR 4: optimizer.step() called directly instead of scaler.step(optimizer)
            optimizer.step()
            
            # ERROR 5: Missing scaler.update() call
            # scaler.update() must be called after scaler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    print("-" * 60)
    print("Training completed!")
    
    # Simple evaluation
    model.eval()
    with torch.no_grad():
        X_test, y_test = generate_synthetic_data(num_samples=200, 
                                                  input_size=input_size, 
                                                  output_size=output_size)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return model

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Train the model
    trained_model = train_model()
    print("\nModel training script finished.")