#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import glob

# Simple Super-Resolution CNN Model
class SuperResolutionNet(nn.Module):
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 12, kernel_size=3, padding=1)  # 12 channels for 2x upscaling (3*2*2)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return x

# Custom Dataset class for loading image pairs
class SRDataset(Dataset):
    def __init__(self, dataset_dir):
        self.lr_images = sorted(glob.glob(os.path.join(dataset_dir, 'lr_*.png')))
        self.hr_images = sorted(glob.glob(os.path.join(dataset_dir, 'hr_*.png')))
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.lr_images)
    
    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_images[idx]).convert('RGB')
        hr_img = Image.open(self.hr_images[idx]).convert('RGB')
        lr_tensor = self.transform(lr_img)
        hr_tensor = self.transform(hr_img)
        return lr_tensor, hr_tensor

def main():
    # Load configuration
    with open('/workspace/config.json', 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = SuperResolutionNet().to(device)
    
    # BUG 2: Learning rate is too small - should be 1e-3 or 1e-4
    # Currently set to 1e-8 which prevents meaningful weight updates
    optimizer = optim.Adam(model.parameters(), lr=1e-8)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Load dataset
    dataset = SRDataset('/workspace/dataset')
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 2), shuffle=True)
    
    print(f"Dataset size: {len(dataset)} image pairs")
    print(f"Starting training for {config.get('epochs', 10)} epochs...")
    
    # Training loop
    num_epochs = config.get('epochs', 10)
    final_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            sr_imgs = model(lr_imgs)
            
            # Calculate loss
            loss = criterion(sr_imgs, hr_imgs)
            
            # BUG 3: optimizer.step() is called BEFORE loss.backward()
            # This means gradients are never computed before the optimizer updates
            optimizer.step()
            loss.backward()
            
            # BUG 1: Zero gradients AFTER backward pass instead of before
            # This causes gradient accumulation across batches
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        final_loss = avg_loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    print(f"\nTraining completed. Final loss: {final_loss:.6f}")
    
    # Save final loss for verification
    with open('/workspace/final_loss.txt', 'w') as f:
        f.write(str(final_loss))

if __name__ == '__main__':
    main()