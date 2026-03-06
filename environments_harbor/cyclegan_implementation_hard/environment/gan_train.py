#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, data_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Hyperparameters
latent_dim = 100
data_dim = 784
batch_size = 64
num_epochs = 20
learning_rate = 0.0002

# Initialize networks
generator = Generator(latent_dim)
discriminator = Discriminator(data_dim)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Save initial generator output for comparison
torch.manual_seed(42)
initial_noise = torch.randn(batch_size, latent_dim)
with torch.no_grad():
    initial_output = generator(initial_noise).clone()

print("Starting GAN training...")

# Training loop
for epoch in range(num_epochs):
    # Generate random training data (simulating real data)
    real_data = torch.randn(batch_size, data_dim) * 0.5 + 0.5
    
    # Labels
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    # ============================================
    # Train Discriminator
    # ============================================
    optimizer_d.zero_grad()
    
    # Real data
    real_output = discriminator(real_data)
    d_loss_real = criterion(real_output, real_labels)
    
    # Fake data
    noise = torch.randn(batch_size, latent_dim)
    fake_data = generator(noise)
    fake_output = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_output, fake_labels)
    
    # Total discriminator loss
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()
    
    # ============================================
    # Train Generator
    # ============================================
    optimizer_g.zero_grad()
    
    # Generate new fake data
    noise = torch.randn(batch_size, latent_dim)
    fake_data = generator(noise)
    
    # Generator tries to fool discriminator
    output = discriminator(fake_data)
    g_loss = criterion(output, real_labels)
    
    # Backpropagation
    g_loss.backward()
    
    # CRITICAL BUG: optimizer_g.step() is NOT called here
    # This means the generator weights are never updated!
    # optimizer_g.step()  # This line should be here but is commented out
    
    # Print losses every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("\nTraining completed!")

# Test if generator learned
torch.manual_seed(42)
final_noise = torch.randn(batch_size, latent_dim)
with torch.no_grad():
    final_output = generator(final_noise)

# Compare initial and final outputs
output_diff = torch.abs(initial_output - final_output).mean().item()
learned = output_diff > 0.1

print(f"\nGenerator output difference: {output_diff:.6f}")
print(f"Generator learned: {learned}")

if not learned:
    print("WARNING: Generator did not learn! Outputs are identical to initialization.")
else:
    print("SUCCESS: Generator shows evidence of learning!")