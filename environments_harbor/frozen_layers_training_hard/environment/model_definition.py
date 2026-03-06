#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Convolutional Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # Input size: 224x224 -> after 4 pooling: 14x14
        # 512 channels * 14 * 14 = 100352
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 224 -> 112
        
        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 112 -> 56
        
        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 56 -> 28
        
        # Convolutional Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)  # 28 -> 14
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


if __name__ == "__main__":
    # Test the model
    model = ConvNet(num_classes=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
```

```python
#!/usr/bin/env python3

import torch
import sys
sys.path.insert(0, '/workspace')

from model_definition import ConvNet


def analyze_and_freeze_layers(model, freeze_ratio=0.60):
    """
    Analyze model layers and freeze the first 60% of parameters by count.
    
    Args:
        model: PyTorch model to analyze
        freeze_ratio: Ratio of parameters to freeze (default 0.60)
    
    Returns:
        Dictionary with freezing statistics
    """
    # Get all parameters with their names
    param_info = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            param_info.append({
                'name': name,
                'param': param,
                'count': param_count
            })
    
    # Calculate total parameters
    total_params = sum(p['count'] for p in param_info)
    target_freeze_count = int(total_params * freeze_ratio)
    
    # Freeze parameters based on cumulative count
    cumulative_count = 0
    frozen_count = 0
    trainable_count = 0
    
    for p_info in param_info:
        if cumulative_count < target_freeze_count:
            # Freeze this parameter
            p_info['param'].requires_grad = False
            frozen_count += p_info['count']
        else:
            # Keep trainable
            p_info['param'].requires_grad = True
            trainable_count += p_info['count']
        
        cumulative_count += p_info['count']
    
    # Calculate actual freeze percentage
    actual_freeze_percentage = frozen_count / total_params if total_params > 0 else 0.0
    
    return {
        'total_parameters': total_params,
        'frozen_parameters': frozen_count,
        'trainable_parameters': trainable_count,
        'freeze_percentage': actual_freeze_percentage
    }


def save_freeze_summary(stats, output_path='/workspace/freeze_summary.txt'):
    """
    Save freezing configuration summary to a file.
    
    Args:
        stats: Dictionary with freezing statistics
        output_path: Path to save the summary
    """
    with open(output_path, 'w') as f:
        f.write(f"total_parameters={stats['total_parameters']}\n")
        f.write(f"frozen_parameters={stats['frozen_parameters']}\n")
        f.write(f"trainable_parameters={stats['trainable_parameters']}\n")
        f.write(f"freeze_percentage={stats['freeze_percentage']:.2f}\n")


def verify_freezing(model):
    """
    Verify that the freezing configuration is correctly applied.
    
    Args:
        model: PyTorch model to verify
    
    Returns:
        Boolean indicating if verification passed
    """
    frozen_count = 0
    trainable_count = 0
    
    for param in model.parameters():
        if param.requires_grad:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
    
    total = frozen_count + trainable_count
    
    print(f"Verification:")
    print(f"  Total parameters: {total:,}")
    print(f"  Frozen parameters: {frozen_count:,}")
    print(f"  Trainable parameters: {trainable_count:,}")
    print(f"  Freeze percentage: {frozen_count/total:.2%}")
    
    return True


def main():
    # Load the model
    print("Loading model...")
    model = ConvNet(num_classes=10)
    
    # Analyze and freeze layers
    print("\nAnalyzing model architecture...")
    stats = analyze_and_freeze_layers(model, freeze_ratio=0.60)
    
    print(f"\nFreezing Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Frozen parameters: {stats['frozen_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  Freeze percentage: {stats['freeze_percentage']:.2%}")
    
    # Verify freezing
    print("\n" + "="*50)
    verify_freezing(model)
    
    # Save summary
    print("\nSaving freeze summary to /workspace/freeze_summary.txt...")
    save_freeze_summary(stats)
    print("Done!")


if __name__ == "__main__":
    main()