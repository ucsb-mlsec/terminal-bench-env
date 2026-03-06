#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

class CVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, num_classes=4):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Load model
model = CVAE(input_dim=784, latent_dim=20, num_classes=4)
model.load_state_dict(torch.load('/tmp/cvae_model.pth', map_location='cpu'))
model.eval()

# Generate samples for each class
results = []
with torch.no_grad():
    for class_label in range(4):
        # Sample from standard normal
        z = torch.randn(1, 20)
        
        # Create one-hot encoded label
        c = torch.zeros(1, 4)
        c[0, class_label] = 1.0
        
        # Generate image
        output = model.decode(z, c)
        
        # Reshape to 28x28 and denormalize to [0, 255]
        img = output.view(28, 28).numpy() * 255.0
        
        # Calculate mean
        mean_val = np.mean(img)
        results.append(mean_val)

# Write to file
with open('/tmp/cvae_validation.txt', 'w') as f:
    for i, mean_val in enumerate(results):
        f.write(f'class_{i}={mean_val:.1f}\n')