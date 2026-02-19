#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Load data
print("Loading data...")
train_images = np.load('/data/shapes/train_images.npy')
train_labels = np.load('/data/shapes/train_labels.npy')

# Normalize images to [0, 1]
train_images = train_images.astype(np.float32) / 255.0

# Convert to torch tensors
train_images = torch.from_numpy(train_images).float()
train_labels = torch.from_numpy(train_labels).long()

print(f"Loaded {len(train_images)} training images")
print(f"Image shape: {train_images.shape}")
print(f"Labels shape: {train_labels.shape}")

# CVAE Model
class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=4):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # After conv layers: 28x28 -> 14x14 -> 7x7
        # Flattened: 64 * 7 * 7 = 3136
        # Add label info (using raw integer label)
        self.fc_enc = nn.Linear(3136 + 1, 256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder - takes latent vector only, no conditioning
        self.fc_dec = nn.Linear(latent_dim, 256)
        self.fc_dec2 = nn.Linear(256, 64 * 7 * 7)
        
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x, label):
        # x shape: (batch, 28, 28)
        x = x.unsqueeze(1)  # (batch, 1, 28, 28)
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = h.view(h.size(0), -1)  # Flatten
        
        # Concatenate with raw label integer
        label_input = label.unsqueeze(1).float()
        h = torch.cat([h, label_input], dim=1)
        
        h = self.relu(self.fc_enc(h))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        # Incorrect std calculation - using logvar directly instead of exp(0.5*logvar)
        std = logvar
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        # Decoder doesn't receive label conditioning
        h = self.relu(self.fc_dec(z))
        h = self.relu(self.fc_dec2(h))
        h = h.view(h.size(0), 64, 7, 7)
        h = self.relu(self.deconv1(h))
        h = self.sigmoid(self.deconv2(h))
        return h.squeeze(1)  # (batch, 28, 28)
    
    def forward(self, x, label):
        mean, logvar = self.encode(x, label)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

# Loss function
def loss_function(recon_x, x, mean, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence with extremely high weight
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Imbalanced weights - KL term dominates
    return BCE + 1000.0 * KLD

# Training setup
device = torch.device('cpu')
model = CVAE(latent_dim=20, num_classes=4).to(device)

# Very high learning rate
optimizer = optim.Adam(model.parameters(), lr=0.1)

batch_size = 32
num_epochs = 80

# Create data loader
dataset = torch.utils.data.TensorDataset(train_images, train_labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
print("Starting training...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        recon, mean, logvar = model(data, labels)
        loss = loss_function(recon, data, mean, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_images)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save model
print("Saving model...")
torch.save(model.state_dict(), '/tmp/cvae_model.pth')
print("Training complete!")