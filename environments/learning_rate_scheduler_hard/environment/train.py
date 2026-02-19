#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import json
import os
from pathlib import Path

# Simple CNN Model for Image Classification
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

# Dummy Dataset for Testing
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        label = torch.randint(0, 10, (1,)).item()
        return image, label

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using default configuration.")
        return {
            'num_epochs': 100,
            'batch_size': 32,
            'initial_lr': 0.001,
            'warmup_epochs': 0,
            'peak_lr': 0.001,
            'final_lr': 0.0001,
            'scheduler_type': 'none',
            'scheduler_params': {}
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

def get_lr_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler based on config"""
    scheduler_type = config.get('scheduler_type', 'none')
    scheduler_params = config.get('scheduler_params', {})
    
    if scheduler_type == 'cosine_annealing':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', 90),
            eta_min=scheduler_params.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_params.get('gamma', 0.95)
        )
    elif scheduler_type == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_params.get('milestones', [30, 60, 90]),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    else:
        return None

def warmup_lr(optimizer, current_epoch, warmup_epochs, initial_lr, peak_lr):
    """Apply linear warmup to learning rate"""
    if current_epoch < warmup_epochs:
        lr = initial_lr + (peak_lr - initial_lr) * (current_epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    return None

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def save_training_log(log_data, log_path):
    """Save training history to JSON file"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config_path = '/workspace/config.yaml'
    config = load_config(config_path)
    
    # Extract hyperparameters
    num_epochs = config.get('num_epochs', 100)
    batch_size = config.get('batch_size', 32)
    initial_lr = config.get('initial_lr', 0.0001)
    peak_lr = config.get('peak_lr', 0.001)
    warmup_epochs = config.get('warmup_epochs', 0)
    
    print(f"Training configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Peak LR: {peak_lr}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Scheduler: {config.get('scheduler_type', 'none')}")
    
    # Create datasets and dataloaders
    train_dataset = DummyDataset(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize model, loss, and optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Initialize scheduler
    steps_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, config, steps_per_epoch)
    
    # Training history
    training_history = {
        'epochs': [],
        'loss': [],
        'accuracy': [],
        'learning_rate': []
    }
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Apply warmup if in warmup phase
        current_lr = None
        if epoch < warmup_epochs:
            current_lr = warmup_lr(optimizer, epoch, warmup_epochs, initial_lr, peak_lr)
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Train for one epoch
        avg_loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Apply scheduler after warmup
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        training_history['epochs'].append(epoch + 1)
        training_history['loss'].append(avg_loss)
        training_history['accuracy'].append(accuracy)
        training_history['learning_rate'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, "
                  f"Acc: {accuracy:.2f}%, LR: {current_lr:.6f}")
    
    # Save training history
    log_path = '/workspace/logs/training_history.json'
    save_training_log(training_history, log_path)
    print(f"\nTraining completed. Logs saved to {log_path}")

if __name__ == '__main__':
    main()