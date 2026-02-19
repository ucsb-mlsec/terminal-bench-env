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
```

Now, here's the CutMix solution:

```python
import torch
import numpy as np


def rand_bbox(size, lam):
    """
    Generate random bounding box coordinates for CutMix.
    
    Args:
        size: Tuple of (batch_size, channels, height, width)
        lam: Lambda value from Beta distribution (controls cut ratio)
    
    Returns:
        Tuple of (bbx1, bby1, bbx2, bby2) - bounding box coordinates
    """
    W = size[2]
    H = size[3]
    
    # Calculate cut ratio based on lambda
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sampling of center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Calculate bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def apply_cutmix(images, labels, alpha=1.0, prob=1.0):
    """
    Apply CutMix augmentation to a batch of images and labels.
    
    CutMix cuts a rectangular patch from one image and pastes it onto another,
    mixing the labels proportionally based on the area of the patch.
    
    Args:
        images: Tensor of shape (N, C, H, W) - batch of images
        labels: Tensor of shape (N,) - class labels for each image
        alpha: Float - parameter for Beta distribution (controls mixing ratio)
               Higher alpha = more uniform mixing, lower alpha = less mixing
        prob: Float - probability of applying CutMix (default 1.0)
    
    Returns:
        mixed_images: Tensor of shape (N, C, H, W) - augmented images
        mixed_labels: Tuple of (labels_a, labels_b, lambda_value)
                     - labels_a: original labels
                     - labels_b: shuffled labels
                     - lambda_value: mixing coefficient
    """
    
    # Check if we should apply CutMix
    if np.random.rand() > prob:
        # Return original data if not applying CutMix
        return images, (labels, labels, 1.0)
    
    batch_size = images.size(0)
    
    # Sample lambda from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # Generate random permutation for mixing
    rand_index = torch.randperm(batch_size).to(images.device)
    
    # Get original and shuffled labels
    labels_a = labels
    labels_b = labels[rand_index]
    
    # Generate bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    
    # Apply CutMix to images
    mixed_images = images.clone()
    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match the box area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    return mixed_images, (labels_a, labels_b, lam)


def cutmix_criterion(criterion, outputs, labels_a, labels_b, lam):
    """
    Compute mixed loss for CutMix augmentation.
    
    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        outputs: Model predictions
        labels_a: Original labels
        labels_b: Mixed labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss value
    """
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


# Example usage function
def example_usage():
    """
    Demonstrate how to use CutMix in a training loop.
    """
    import torch.nn as nn
    
    # Simulate a batch of images and labels
    batch_size = 16
    num_classes = 10
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print("Original batch shape:", images.shape)
    print("Original labels shape:", labels.shape)
    
    # Apply CutMix
    mixed_images, (labels_a, labels_b, lam) = apply_cutmix(images, labels, alpha=1.0)
    
    print("\nAfter CutMix:")
    print("Mixed images shape:", mixed_images.shape)
    print("Lambda value:", lam)
    print("Labels A:", labels_a)
    print("Labels B:", labels_b)
    
    # Example of computing loss with CutMix
    criterion = nn.CrossEntropyLoss()
    dummy_outputs = torch.randn(batch_size, num_classes)
    
    loss = cutmix_criterion(criterion, dummy_outputs, labels_a, labels_b, lam)
    print("\nMixed loss:", loss.item())


if __name__ == '__main__':
    example_usage()