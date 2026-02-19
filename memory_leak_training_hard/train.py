#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Simple CNN model for image classification
class SimpleCNN(nn.Module):
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

def generate_batch(batch_size=64, image_size=32, num_classes=10):
    """Generate synthetic training data"""
    images = torch.randn(batch_size, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels

def train():
    print("Starting training script...")
    print("Initializing model and optimizer...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training configuration
    batch_size = 64
    num_iterations = 1000
    print_every = 10
    
    # This list will cause the memory leak!
    # Storing loss tensors with computation graph attached
    loss_history = []
    
    print(f"\nStarting training loop for {num_iterations} iterations...")
    print(f"Batch size: {batch_size}")
    print("-" * 60)
    
    start_time = time.time()
    
    for iteration in range(1, num_iterations + 1):
        # Generate synthetic data
        images, labels = generate_batch(batch_size=batch_size)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Memory leak: storing the full loss tensor with gradients
        # This keeps the entire computation graph in memory
        loss_history.append(loss)
        
        # Print progress
        if iteration % print_every == 0:
            elapsed = time.time() - start_time
            avg_loss = sum([l.item() for l in loss_history[-print_every:]]) / min(print_every, len(loss_history))
            print(f"Iteration [{iteration}/{num_iterations}] | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
        # This will eventually cause memory issues
        if iteration % 50 == 0:
            print(f"  -> Loss history size: {len(loss_history)} items")
    
    print("-" * 60)
    print(f"Training completed! Total time: {time.time() - start_time:.2f}s")
    print(f"Final average loss: {sum([l.item() for l in loss_history[-100:]]) / 100:.4f}")

if __name__ == "__main__":
    try:
        train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 60)
            print("ERROR: Out of memory!")
            print("=" * 60)
            print("The training script crashed due to memory issues.")
        else:
            raise e