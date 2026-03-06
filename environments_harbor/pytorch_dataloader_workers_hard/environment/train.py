#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import time

class ImageDataset(Dataset):
    """Custom Dataset for loading images from a directory"""
    def __init__(self, directory):
        self.directory = directory
        self.image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(self.image_files) == 0:
            # If no actual images, create dummy file list for testing
            self.image_files = [f'dummy_{i}.jpg' for i in range(200)]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Return dummy tensor data for demonstration
        # In real scenario, this would load actual images
        image = torch.randn(3, 224, 224)
        label = idx % 10  # Dummy labels 0-9
        return image, label


class SimpleModel(nn.Module):
    """Simple CNN model for image classification"""
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
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


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset
    dataset_path = '/workspace/dataset/'
    dataset = ImageDataset(dataset_path)
    print(f"Dataset size: {len(dataset)} images")
    
    # PROBLEMATIC DataLoader configuration
    # Training seems slow - GPU often waits for data
    # CPU cores are not being fully utilized
    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Fixed batch size
        shuffle=True,
        num_workers=0,  # PROBLEM: No parallel data loading! Single-threaded!
        pin_memory=False,  # PROBLEM: Inefficient CPU-to-GPU memory transfers
        # persistent_workers not set - PROBLEM: Workers respawn each epoch (if num_workers > 0)
    )
    
    # Initialize model, loss, and optimizer
    model = SimpleModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move data to device
            # Note: This transfer is slow due to pin_memory=False
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {avg_loss:.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = running_loss / len(dataloader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print("=" * 60)
        
        # Performance note: Epoch times are inconsistent and slower than expected
        # GPU utilization monitoring shows frequent idle periods
        # CPU monitoring shows underutilization - only 1 core actively loading data
    
    print("\nTraining completed!")
    print("Note: Training performance is suboptimal due to DataLoader configuration")
    print("Consider investigating num_workers, pin_memory, and persistent_workers settings")


if __name__ == '__main__':
    main()