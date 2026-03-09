import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling operations: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input: NCHW format (batch, channels, height, width)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def get_data_loaders(data_dir='/workspace/data/', batch_size=16):
    """Setup data loaders with basic transforms"""
    
    # Basic transforms only - NO augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    if os.path.exists(data_dir):
        train_dataset = ImageFolder(root=data_dir, transform=transform)
        train_loader = DataLoader(train_dataset, 
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=2)
        return train_loader
    else:
        print(f"Warning: Data directory {data_dir} not found!")
        return None


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Training loop"""
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # TODO: Add CutMix augmentation here for better generalization
            # This would help reduce overfitting and improve model generalization
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 9:
                avg_loss = running_loss / 10
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}], '
                      f'Loss: {avg_loss:.4f}')
                running_loss = 0.0
        
        print(f'Epoch [{epoch+1}/{num_epochs}] completed')


def main():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    num_classes = 10
    batch_size = 16
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 10
    
    # Initialize model
    model = SimpleCNN(num_classes=num_classes).to(device)
    print('Model initialized')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                         lr=learning_rate, 
                         momentum=momentum)
    
    # Get data loader
    train_loader = get_data_loaders(data_dir='/workspace/data/', 
                                   batch_size=batch_size)
    
    if train_loader is None:
        print("Creating dummy data for demonstration...")
        # Create dummy dataset for testing if data directory doesn't exist
        dummy_images = torch.randn(64, 3, 224, 224)
        dummy_labels = torch.randint(0, num_classes, (64,))
        dummy_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
        train_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)
    
    print(f'Starting training for {num_epochs} epochs...')
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    
    print('Training completed!')
    
    # Save the model
    model_path = '/workspace/model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()