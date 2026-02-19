#!/usr/bin/env python3
"""
Training script for image classification model
This script implements mixed precision training using torch.cuda.amp
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleCNN(nn.Module):
    """
    Simple CNN model for image classification
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_dummy_dataset(num_samples=1000, batch_size=32):
    """
    Create a dummy dataset similar to CIFAR-10 format
    Images: 32x32x3, Labels: 10 classes
    """
    # Generate random images and labels
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """
    Train for one epoch with mixed precision training
    Note: This function is supposed to use torch.cuda.amp for mixed precision
    """
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass - ISSUE 2: Missing autocast context manager
        # Should be wrapped in: with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass - ISSUE 1: Not using scaler.scale() for loss
        # Should be: scaler.scale(loss).backward()
        loss.backward()
        
        # ISSUE 4: Gradient clipping on scaled gradients
        # This should be done after scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step - ISSUE 3: Not using scaler.step() and scaler.update()
        # Should be: scaler.step(optimizer) followed by scaler.update()
        optimizer.step()
        
        # Calculate metrics
        accuracy = calculate_accuracy(outputs, labels)
        running_loss += loss.item()
        running_accuracy += accuracy
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def train_model(model, train_loader, criterion, optimizer, scaler, device, num_epochs=10):
    """
    Main training loop with mixed precision support
    """
    print(f"Training on device: {device}")
    print(f"Using mixed precision training: {torch.cuda.is_available()}")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        epoch_loss, epoch_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Average Accuracy: {epoch_accuracy:.2f}%")
        print("-" * 60)
    
    print("\nTraining completed!")


def main():
    """
    Main function to set up and run training
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_classes = 10
    
    # Create model
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize GradScaler for mixed precision training
    # The scaler is created but not properly used in the training loop
    scaler = torch.cuda.amp.GradScaler()
    
    # Create dataset
    print("\nCreating dataset...")
    train_loader = create_dummy_dataset(num_samples=1000, batch_size=batch_size)
    print(f"Dataset created with {len(train_loader.dataset)} samples")
    
    # Train the model
    print("\nStarting training with mixed precision (AMP)...")
    train_model(model, train_loader, criterion, optimizer, scaler, device, num_epochs)
    
    # Save the trained model
    model_path = '/workspace/model_checkpoint.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()