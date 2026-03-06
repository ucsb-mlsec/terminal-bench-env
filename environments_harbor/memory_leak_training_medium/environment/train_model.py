#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import glob

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_dataset(data_dir):
    """Load images from directory and create dataset"""
    image_files = glob.glob(os.path.join(data_dir, '*.png')) + \
                  glob.glob(os.path.join(data_dir, '*.jpg')) + \
                  glob.glob(os.path.join(data_dir, '*.jpeg'))
    
    dataset = []
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    np.random.seed(42)
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img_tensor = transform(img)
            label = np.random.randint(0, 10)
            dataset.append((img_tensor, label))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return dataset

def main():
    print("Initializing training...")
    
    # Load dataset
    data_dir = '/workspace/data/'
    dataset = load_dataset(data_dir)
    print(f"Loaded {len(dataset)} images")
    
    # Initialize model, optimizer, and loss function
    model = ImageClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    num_epochs = 10
    batch_size = 8
    
    # MEMORY LEAK: List to store loss history (will store tensors with gradients)
    loss_history = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # MEMORY LEAK: Accumulating outputs across batches
        batch_outputs = []
        
        # Create batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if len(batch) == 0:
                continue
                
            # Prepare batch
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # MEMORY LEAK: Appending loss tensor directly without detaching
            loss_history.append(loss)
            
            # MEMORY LEAK: Accumulating model outputs with gradients
            batch_outputs.append(outputs)
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if epoch % 2 == 0:
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    print(f"Total losses recorded: {len(loss_history)}")

if __name__ == '__main__':
    main()