#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
from pathlib import Path
from model import SimpleModel

def load_data(filepath):
    """Load data from CSV file."""
    df = pd.read_csv(filepath)
    X = torch.FloatTensor(df['x'].values).unsqueeze(1)
    y = torch.FloatTensor(df['y'].values).unsqueeze(1)
    return X, y

def train_epoch(model, optimizer, criterion, X, y):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, X, y, criterion):
    """Evaluate model on given data."""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = criterion(predictions, y).item()
    return mse

def main():
    # Initialize model, optimizer, and loss function
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Define data directory
    data_dir = Path('/workspace/continual_learning/data')
    
    # Load all training data
    X1_train, y1_train = load_data(data_dir / 'batch1_train.csv')
    X2_train, y2_train = load_data(data_dir / 'batch2_train.csv')
    X3_train, y3_train = load_data(data_dir / 'batch3_train.csv')
    
    # Load all test data
    X1_test, y1_test = load_data(data_dir / 'batch1_test.csv')
    X2_test, y2_test = load_data(data_dir / 'batch2_test.csv')
    X3_test, y3_test = load_data(data_dir / 'batch3_test.csv')
    
    # Train on Batch 1
    print("Training on Batch 1...")
    for epoch in range(50):
        loss = train_epoch(model, optimizer, criterion, X1_train, y1_train)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50, Loss: {loss:.4f}")
    
    # Train on Batch 2
    print("\nTraining on Batch 2...")
    for epoch in range(50):
        loss = train_epoch(model, optimizer, criterion, X2_train, y2_train)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50, Loss: {loss:.4f}")
    
    # Train on Batch 3
    print("\nTraining on Batch 3...")
    for epoch in range(50):
        loss = train_epoch(model, optimizer, criterion, X3_train, y3_train)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50, Loss: {loss:.4f}")
    
    # Evaluate on all test sets
    print("\nEvaluating on all test sets...")
    batch1_mse = evaluate(model, X1_test, y1_test, criterion)
    batch2_mse = evaluate(model, X2_test, y2_test, criterion)
    batch3_mse = evaluate(model, X3_test, y3_test, criterion)
    average_mse = (batch1_mse + batch2_mse + batch3_mse) / 3
    
    print(f"\nBatch 1 MSE: {batch1_mse:.4f}")
    print(f"Batch 2 MSE: {batch2_mse:.4f}")
    print(f"Batch 3 MSE: {batch3_mse:.4f}")
    print(f"Average MSE: {average_mse:.4f}")
    
    # Save results to JSON
    results = {
        "batch1_mse": float(batch1_mse),
        "batch2_mse": float(batch2_mse),
        "batch3_mse": float(batch3_mse),
        "average_mse": float(average_mse)
    }
    
    output_path = Path('/workspace/solution.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Check if performance target is met
    if batch1_mse < 2.0 and batch2_mse < 2.0 and batch3_mse < 2.0 and average_mse < 2.0:
        print("\n✓ Performance target met!")
    else:
        print("\n✗ Performance target not met. MSE values should be below 2.0")

if __name__ == "__main__":
    main()