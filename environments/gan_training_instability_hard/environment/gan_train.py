#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Generator class (COMPLETE - do not modify)
class Generator(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator class (COMPLETE - do not modify)
class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Data generation function (COMPLETE - do not modify)
def generate_data(n_samples):
    """Generate 2D mixture of Gaussians with 8 modes arranged in a circle"""
    n_modes = 8
    samples_per_mode = n_samples // n_modes
    data = []
    
    for i in range(n_modes):
        angle = 2 * np.pi * i / n_modes
        center_x = 2.0 * np.cos(angle)
        center_y = 2.0 * np.sin(angle)
        
        samples = np.random.randn(samples_per_mode, 2) * 0.05
        samples[:, 0] += center_x
        samples[:, 1] += center_y
        data.append(samples)
    
    data = np.vstack(data)
    np.random.shuffle(data)
    return data[:n_samples]

# Gradient penalty function (INCOMPLETE - marked with # TODO: Implement this)
def compute_gradient_penalty(discriminator, real_data, fake_data, device):
    # TODO: Implement this
    pass

# Train discriminator function (INCOMPLETE - marked with # TODO: Implement this)
def train_discriminator(discriminator, generator, optimizer_d, real_data, latent_dim, device, lambda_gp=10):
    # TODO: Implement this
    pass

# Train generator function (INCOMPLETE - marked with # TODO: Implement this)
def train_generator(generator, discriminator, optimizer_g, batch_size, latent_dim, device):
    # TODO: Implement this
    pass

# Main training code (COMPLETE - do not modify)
if __name__ == "__main__":
    # Set device
    device = 'cpu'
    
    # Hyperparameters
    latent_dim = 2
    hidden_dim = 128
    batch_size = 256
    n_iterations = 1000
    n_critic = 5
    learning_rate = 1e-4
    
    # Initialize models
    generator = Generator(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    discriminator = Discriminator(input_dim=2, hidden_dim=hidden_dim).to(device)
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Training loop
    try:
        for iteration in range(n_iterations):
            # Train discriminator
            for _ in range(n_critic):
                real_data = generate_data(batch_size)
                real_data = torch.FloatTensor(real_data).to(device)
                
                d_loss = train_discriminator(discriminator, generator, optimizer_d, real_data, latent_dim, device)
            
            # Train generator
            g_loss = train_generator(generator, discriminator, optimizer_g, batch_size, latent_dim, device)
            
            # Check for NaN or Inf
            if not np.isfinite(d_loss) or not np.isfinite(g_loss):
                raise ValueError(f"Loss became non-finite at iteration {iteration}")
            
            # Log progress
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        # Training completed successfully
        result = {
            "status": "success",
            "final_iteration": n_iterations
        }
        
        with open('/workspace/training_result.json', 'w') as f:
            json.dump(result, f)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        result = {
            "status": "failed",
            "error": str(e)
        }
        with open('/workspace/training_result.json', 'w') as f:
            json.dump(result, f)