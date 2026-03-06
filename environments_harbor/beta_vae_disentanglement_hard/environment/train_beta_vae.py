#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ShapeDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        return image


def loss_function(recon_x, x, mu, logvar, beta):
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = recon_loss / x.size(0)  # Average over batch
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch
    
    # INTENTIONAL BUG: Incorrect beta application
    # The bug is that beta is applied to reconstruction loss instead of KL divergence
    # Correct formula should be: loss = recon_loss + beta * kl_loss
    # But we're implementing: loss = beta * recon_loss + kl_loss
    total_loss = beta * recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss


def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config


def load_dataset(data_dir, train_split=0.8):
    data_path = Path(data_dir)
    image_files = sorted(list(data_path.glob('*.png')) + list(data_path.glob('*.jpg')))
    
    # Split into train and validation
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    train_dataset = ShapeDataset(train_files)
    val_dataset = ShapeDataset(val_files)
    
    return train_dataset, val_dataset


def train_epoch(model, dataloader, optimizer, beta, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def main():
    # Load configuration
    config = load_config()
    learning_rate = config['learning_rate']
    beta = config['beta']
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset, val_dataset = load_dataset('/workspace/dataset/', train_split=0.8)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = BetaVAE(latent_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = 50
    final_recon_loss = 0
    final_kl_loss = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Beta parameter: {beta}")
    print(f"Learning rate: {learning_rate}\n")
    
    for epoch in range(1, num_epochs + 1):
        avg_loss, avg_recon_loss, avg_kl_loss = train_epoch(
            model, train_loader, optimizer, beta, device
        )
        
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Loss: {avg_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"KL Div: {avg_kl_loss:.4f}")
        
        # Save final metrics from the last epoch
        if epoch == num_epochs:
            final_recon_loss = avg_recon_loss
            final_kl_loss = avg_kl_loss
    
    # Save results
    results = {
        "final_reconstruction_loss": float(final_recon_loss),
        "final_kl_divergence": float(final_kl_loss),
        "epochs_completed": num_epochs
    }
    
    os.makedirs('/tmp', exist_ok=True)
    with open('/tmp/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Results saved to /tmp/training_results.json")
    print(f"Final Reconstruction Loss: {final_recon_loss:.4f}")
    print(f"Final KL Divergence: {final_kl_loss:.4f}")


if __name__ == '__main__':
    main()