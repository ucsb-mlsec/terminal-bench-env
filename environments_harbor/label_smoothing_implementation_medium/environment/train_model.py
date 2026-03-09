#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class SimpleCNN(nn.Module):
    """Simple CNN for 10-class image classification"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ImageDataset(Dataset):
    """Custom Dataset that wraps labels with dummy image data"""
    def __init__(self, labels, image_size=(3, 32, 32)):
        self.labels = labels
        self.image_size = image_size
        # Generate dummy images for demonstration
        np.random.seed(42)
        self.images = np.random.randn(len(labels), *image_size).astype(np.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        # NOTE: Using hard one-hot labels directly - this leads to overconfident predictions
        # Label smoothing should be applied here to improve calibration
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Using hard labels leads to overconfident predictions
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        _, true_labels = labels.max(1)
        total += labels.size(0)
        correct += predicted.eq(true_labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            _, true_labels = labels.max(1)
            total += labels.size(0)
            correct += predicted.eq(true_labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load hard one-hot encoded labels
    # TODO: Apply label smoothing to these labels before training
    train_labels = np.load('/workspace/data/train_labels.npy')
    val_labels = np.load('/workspace/data/val_labels.npy')
    
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    
    # Create datasets
    train_dataset = ImageDataset(train_labels)
    val_dataset = ImageDataset(val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = SimpleCNN(num_classes=10).to(device)
    
    # Loss function - CrossEntropyLoss works with soft labels
    # but currently receiving hard one-hot labels causing overconfidence
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    print("\nStarting training with hard one-hot labels...")
    print("NOTE: Model may show overconfident predictions and poor calibration\n")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("\nTraining complete!")
    print("Consider implementing label smoothing to improve model calibration.")


if __name__ == '__main__':
    # Create data directory and generate dummy data if not exists
    os.makedirs('/workspace/data', exist_ok=True)

    if not os.path.exists('/workspace/data/train_labels.npy'):
        print("Generating dummy training labels...")
        train_labels = np.eye(10)[np.random.randint(0, 10, 5000)]
        np.save('/workspace/data/train_labels.npy', train_labels)

    if not os.path.exists('/workspace/data/val_labels.npy'):
        print("Generating dummy validation labels...")
        val_labels = np.eye(10)[np.random.randint(0, 10, 1000)]
        np.save('/workspace/data/val_labels.npy', val_labels)

    main()