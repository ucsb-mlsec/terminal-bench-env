#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generator network
class Generator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim=2):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Gradient penalty calculation
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Load training data
train_data = np.load('/workspace/train_data.npy')
train_data = torch.FloatTensor(train_data)

# Hyperparameters (MISCONFIGURED)
gen_lr = 0.01  # Learning rate for generator (TOO HIGH)
disc_lr = 0.0001  # Learning rate for discriminator (TOO LOW - imbalanced)
lambda_gp = 0.1  # Gradient penalty coefficient (TOO LOW - should be ~10)
n_critic = 1  # Discriminator updates per generator update (TOO LOW - should be ~5)
batch_size = 512  # Batch size (TOO LARGE for 1000 samples)
n_iterations = 500
noise_dim = 2

# Initialize networks
generator = Generator(noise_dim=noise_dim, output_dim=2)
discriminator = Discriminator(input_dim=2)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))

# Training loop
print("Starting WGAN-GP training...")
for iteration in range(n_iterations):
    
    # Train Discriminator
    for _ in range(n_critic):
        optimizer_d.zero_grad()
        
        # Sample real data
        indices = np.random.choice(len(train_data), batch_size, replace=True)
        real_samples = train_data[indices]
        
        # Generate fake data
        noise = torch.randn(batch_size, noise_dim)
        fake_samples = generator(noise).detach()
        
        # Discriminator scores
        real_scores = discriminator(real_samples)
        fake_scores = discriminator(fake_samples)
        
        # Compute gradient penalty
        gp = compute_gradient_penalty(discriminator, real_samples, fake_samples)
        
        # Wasserstein loss with gradient penalty
        d_loss = fake_scores.mean() - real_scores.mean() + lambda_gp * gp
        
        d_loss.backward()
        optimizer_d.step()
    
    # Train Generator
    optimizer_g.zero_grad()
    
    noise = torch.randn(batch_size, noise_dim)
    fake_samples = generator(noise)
    fake_scores = discriminator(fake_samples)
    
    g_loss = -fake_scores.mean()
    
    g_loss.backward()
    optimizer_g.step()
    
    # Print progress
    if (iteration + 1) % 50 == 0:
        print(f"Iteration {iteration + 1}/{n_iterations} | "
              f"D Loss: {d_loss.item():.4f} | "
              f"G Loss: {g_loss.item():.4f} | "
              f"GP: {gp.item():.4f}")

# Save final metrics
final_metrics = {
    'disc_loss': d_loss.item(),
    'gen_loss': g_loss.item(),
    'gp_value': gp.item()
}

with open('/workspace/metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)

print("Training completed. Metrics saved to /workspace/metrics.json")