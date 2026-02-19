#!/usr/bin/env python3

import ray
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from model import NeuralNetwork

def load_data(train_path, val_path):
    """Load training and validation data from CSV files."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Separate features and labels
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values
    y_val = val_df.iloc[:, -1].values
    
    return X_train, y_train, X_val, y_val

@ray.remote
def train_model_distributed(X_train, y_train, X_val, y_val, epochs=100, lr=0.01):
    """Distributed training function that trains the neural network."""
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)  # BUG 1: Should be LongTensor for classification
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)  # BUG 1: Should be LongTensor for classification
    
    # Initialize model
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # BUG 2: Should use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # BUG 3: Missing optimizer.step() - weights never update!
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, predicted = torch.max(val_outputs, 1)
        accuracy = (predicted == y_val_tensor).float().mean().item()
    
    return accuracy

def main():
    """Main function to orchestrate distributed training."""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data(
        '/workspace/data_train.csv',
        '/workspace/data_val.csv'
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Launch distributed training
    print("Starting distributed training...")
    future = train_model_distributed.remote(
        X_train, y_train, X_val, y_val,
        epochs=100,
        lr=0.01
    )
    
    # Get the result
    validation_accuracy = ray.get(future)
    
    print(f"\nTraining completed!")
    print(f"Final validation accuracy: {validation_accuracy:.4f}")
    
    # Determine if converged
    converged = validation_accuracy >= 0.90
    
    # Save results
    results = {
        "validation_accuracy": round(validation_accuracy, 4),
        "converged": converged
    }
    
    with open('/workspace/result.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to /workspace/result.json")
    print(f"Converged: {converged}")
    
    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()