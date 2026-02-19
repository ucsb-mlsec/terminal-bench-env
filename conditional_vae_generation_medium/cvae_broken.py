#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder - BUG: expecting input_dim + num_classes but will receive wrong concatenation
        self.fc1 = nn.Linear(input_dim + num_classes, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder - BUG: will receive wrong concatenation of z and labels
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, input_dim)
        
    def encode(self, x, labels):
        # BUG 1: Not one-hot encoding labels properly
        # labels are just integers, not one-hot vectors
        # BUG 2: Concatenating along wrong dimension
        h = torch.cat([x, labels], dim=0)  # Wrong dimension!
        h = F.relu(self.fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # BUG: Incorrect dimension handling in reparameterization
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(mu)
        # BUG: Adding dimension mismatch
        z = mu + std * eps.unsqueeze(0)  # Wrong unsqueeze!
        return z
    
    def decode(self, z, labels):
        # BUG: labels not properly processed
        # BUG: Missing proper one-hot encoding
        # BUG: Wrong concatenation
        h = torch.cat([z, labels.float()], dim=1)  # labels not one-hot!
        h = F.relu(self.fc3(h))
        reconstruction = torch.sigmoid(self.fc4(h))
        return reconstruction
    
    def forward(self, x, labels):
        # BUG: Labels passed directly as integers instead of one-hot
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels)
        return reconstruction, mu, logvar