#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import os

# Import local modules
from model import BinaryClassifier
from data_generator import generate_data

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, optimizer, criterion, data, labels, batch_size):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Create batches
    num_samples = data.size(0)
    indices = torch.randperm(num_samples)
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_samples)]
        batch_data = data[batch_indices]
        batch_labels = labels[batch_indices]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs.squeeze(), batch_labels)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {num_batches}")
            return None
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def main():
    """Main training function"""
    print("Starting training pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    
    # Get training parameters
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    num_samples = config['data']['num_samples']
    
    print(f"Configuration loaded: {num_epochs} epochs, batch size {batch_size}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data, labels = generate_data(num_samples, input_dim)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Initialize model
    print("Initializing model...")
    model = BinaryClassifier(input_dim, hidden_dim)
    print(f"Model architecture:\n{model}")
    
    # Set up optimizer with EXTREMELY HIGH learning rate (BUG!)
    learning_rate = 100.0  # This is way too high and will cause NaN losses!
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up loss function
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\nStarting training with learning rate: {learning_rate}")
    print("-" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, optimizer, criterion, data, labels, batch_size)
        
        if epoch_loss is None:
            print(f"\nTraining failed at epoch {epoch + 1} due to NaN loss!")
            print("This typically indicates:")
            print("  - Learning rate is too high")
            print("  - Gradient explosion")
            print("  - Numerical instability")
            sys.exit(1)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
        # Early stopping if loss is too high or unstable
        if epoch_loss > 1e6:
            print(f"\nLoss exploded to {epoch_loss}! Training unstable.")
            sys.exit(1)
    
    print("-" * 60)
    print("Training completed successfully")
    
    # Evaluate final model
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        final_loss = criterion(outputs.squeeze(), labels)
        print(f"Final loss: {final_loss.item():.6f}")
        
        # Calculate accuracy
        predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        print(f"Final accuracy: {accuracy.item():.4f}")

if __name__ == "__main__":
    main()