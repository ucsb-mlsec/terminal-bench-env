#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
import os

class VAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(784, 400)
        self.fc_mean = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        
        self.latent_dim = latent_dim
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # Problematic sigmoid activation
    
    def forward(self, x):
        mean, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def vae_loss(recon_x, x, mean, logvar, beta=1.0):
    # Using MSE for reconstruction loss (poor choice)
    recon_loss = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Total loss with beta weight (beta=1.0 is too high)
    return recon_loss + beta * kl_div

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mean, logvar = model(data)
        loss = vae_loss(recon_batch, data, mean, logvar, beta=1.0)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}: Average loss: {avg_loss:.4f}')
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mean, logvar = model(data)
            test_loss += vae_loss(recon_batch, data, mean, logvar, beta=1.0).item()
    
    avg_loss = test_loss / len(test_loader.dataset)
    return avg_loss

def save_reconstructions(model, device, test_loader, filename='/workspace/reconstructions.png'):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:10].to(device)
        recon_batch, _, _ = model(data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            # Original images
            axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstructed images
            axes[1, i].imshow(recon_batch[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 128
    epochs = 10
    latent_dim = 10  # Too small for good reconstruction
    learning_rate = 1e-3
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    final_loss = 0
    for epoch in range(1, epochs + 1):
        final_loss = train(model, device, train_loader, optimizer, epoch)
    
    # Test final loss
    test_loss = test(model, device, test_loader)
    print(f'\nFinal Test Loss: {test_loss:.4f}')
    
    # Generate reconstructions
    os.makedirs('/workspace', exist_ok=True)
    save_reconstructions(model, device, test_loader)
    
    # Save results
    results = {
        "final_loss": final_loss,
        "epochs_trained": epochs,
        "latent_dim": latent_dim,
        "reconstruction_quality": "improved" if final_loss < 100 else "poor"
    }
    
    with open('/workspace/vae_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to /workspace/vae_results.json")
    print(f"Reconstruction quality: {results['reconstruction_quality']}")

if __name__ == '__main__':
    main()