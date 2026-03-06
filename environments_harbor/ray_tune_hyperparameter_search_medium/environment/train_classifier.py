#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, img_size=32):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Generate synthetic images with patterns based on class
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            label = i % num_classes
            # Create images with class-dependent patterns
            img = np.random.randn(3, img_size, img_size).astype(np.float32) * 0.3
            
            # Add class-specific patterns to make the task learnable
            if label < 5:
                # Add horizontal patterns for first 5 classes
                img[0, :, :] += label * 0.5
                img[:, label*6:(label+1)*6, :] += 1.0
            else:
                # Add vertical patterns for last 5 classes
                img[1, :, :] += (label - 5) * 0.5
                img[:, :, (label-5)*6:(label-4)*6] += 1.0
            
            # Add some noise correlation with label
            img += np.random.randn(3, img_size, img_size).astype(np.float32) * 0.1 * (1 + label * 0.05)
            
            self.images.append(img)
            self.labels.append(label)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.images[idx]), torch.LongTensor([self.labels[idx]])[0]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_model(lr, batch_size, dropout, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate datasets
    train_dataset = SyntheticImageDataset(num_samples=1500, num_classes=10)
    val_dataset = SyntheticImageDataset(num_samples=300, num_classes=10)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SimpleCNN(num_classes=10, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    return val_accuracy

def main():
    parser = argparse.ArgumentParser(description='Train image classifier')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    args = parser.parse_args()
    
    val_accuracy = train_model(args.lr, args.batch_size, args.dropout)
    
    print(f'val_accuracy: {val_accuracy:.4f}')

if __name__ == '__main__':
    main()
```

Now I'll create a hyperparameter search script:

```python
#!/usr/bin/env python3

import subprocess
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

def run_training(params):
    """Run training with given hyperparameters and return validation accuracy."""
    lr, batch_size, dropout = params
    
    cmd = [
        'python', '/workspace/train_classifier.py',
        '--lr', str(lr),
        '--batch_size', str(batch_size),
        '--dropout', str(dropout)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        
        # Parse validation accuracy from output
        for line in output.split('\n'):
            if 'val_accuracy:' in line:
                val_acc = float(line.split('val_accuracy:')[1].strip())
                return {
                    'lr': lr,
                    'batch_size': batch_size,
                    'dropout': dropout,
                    'val_accuracy': val_acc
                }
    except Exception as e:
        print(f"Error running training with params {params}: {e}")
        return None
    
    return None

def generate_hyperparameter_grid(n_samples=25):
    """Generate hyperparameter combinations to test."""
    # Use a combination of grid search and random sampling
    
    # Grid search for key values
    lr_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1]
    batch_size_values = [16, 32, 48, 64, 96, 128]
    dropout_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Generate combinations
    np.random.seed(42)
    
    # Start with some good candidates
    good_candidates = [
        (0.05, 64, 0.3),
        (0.03, 64, 0.2),
        (0.05, 48, 0.3),
        (0.07, 64, 0.3),
        (0.05, 96, 0.25),
    ]
    
    # Add grid samples
    grid_samples = list(itertools.product(
        [0.01, 0.05],
        [32, 64],
        [0.3, 0.5]
    ))
    
    # Add random samples
    random_samples = []
    for _ in range(15):
        lr = np.random.choice(lr_values)
        batch_size = np.random.choice(batch_size_values)
        dropout = np.random.choice(dropout_values)
        random_samples.append((lr, batch_size, dropout))
    
    # Combine all samples and remove duplicates
    all_samples = good_candidates + grid_samples + random_samples
    unique_samples = list(set(all_samples))
    
    # Ensure we have at least n_samples
    while len(unique_samples) < n_samples:
        lr = np.random.choice(lr_values)
        batch_size = np.random.choice(batch_size_values)
        dropout = np.random.choice(dropout_values)
        if (lr, batch_size, dropout) not in unique_samples:
            unique_samples.append((lr, batch_size, dropout))
    
    return unique_samples[:n_samples]

def main():
    print("Starting hyperparameter search...")
    print("Generating hyperparameter combinations...")
    
    # Generate at least 20 combinations
    param_combinations = generate_hyperparameter_grid(n_samples=25)
    print(f"Testing {len(param_combinations)} hyperparameter combinations...")
    
    results = []
    
    # Use parallel execution to speed up search
    max_workers = 4  # Adjust based on available CPU cores
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {
            executor.submit(run_training, params): params 
            for params in param_combinations
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_params), 1):
            result = future.result()
            if result:
                results.append(result)
                print(f"[{i}/{len(param_combinations)}] "
                      f"lr={result['lr']}, batch_size={result['batch_size']}, "
                      f"dropout={result['dropout']}, "
                      f"val_accuracy={result['val_accuracy']:.4f}")
    
    if not results:
        print("No successful runs! Check the training script.")
        return
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['val_accuracy'])
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)
    print(f"Total configurations tested: {len(results)}")
    print(f"\nBest configuration:")
    print(f"  Learning rate: {best_result['lr']}")
    print(f"  Batch size: {best_result['batch_size']}")
    print(f"  Dropout: {best_result['dropout']}")
    print(f"  Validation accuracy: {best_result['val_accuracy']:.4f}")
    
    # Save best configuration
    best_config = {
        'lr': float(best_result['lr']),
        'batch_size': int(best_result['batch_size']),
        'dropout': float(best_result['dropout'])
    }
    
    with open('/tmp/best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest configuration saved to /tmp/best_config.json")
    
    # Show top 5 configurations
    print("\nTop 5 configurations:")
    sorted_results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)[:5]
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. lr={result['lr']}, batch_size={result['batch_size']}, "
              f"dropout={result['dropout']}, val_accuracy={result['val_accuracy']:.4f}")

if __name__ == '__main__':
    main()