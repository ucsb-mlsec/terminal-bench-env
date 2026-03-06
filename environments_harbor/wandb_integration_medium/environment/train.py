#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

# Hyperparameters
config = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 10,
    'hidden_size': 128,
    'dropout': 0.2,
    'input_size': 20,
    'num_classes': 3,
    'train_split': 0.8
}

# Simple Neural Network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def generate_synthetic_data(num_samples=2000, input_size=20, num_classes=3):
    """Generate synthetic classification data"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = torch.randn(num_samples, input_size)
    # Create synthetic labels with some pattern
    y = (X[:, :3].sum(dim=1) > 0).long()
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

def calculate_accuracy(outputs, labels):
    """Calculate classification accuracy"""
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels)
        num_batches += 1
    
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches
    return avg_loss, avg_accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, labels)
            num_batches += 1
    
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches
    return avg_loss, avg_accuracy

def train_model(config):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(
        num_samples=2000,
        input_size=config['input_size'],
        num_classes=config['num_classes']
    )
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    # Split into train and validation
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Initialize model
    model = SimpleClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print("\nStarting training...")
    print("=" * 70)
    
    # Training loop
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
    
    print("=" * 70)
    
    # Save model checkpoint
    checkpoint_path = 'model_checkpoint.pth'
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")
    
    return model

if __name__ == "__main__":
    print("Simple Neural Network Training Script")
    print("=" * 70)
    print("\nHyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Train the model
    model = train_model(config)
    
    print("\nTraining completed successfully!")