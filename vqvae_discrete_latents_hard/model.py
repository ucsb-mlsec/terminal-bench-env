#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    VectorQuantizer without EMA updates - THIS IS THE BROKEN VERSION
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embedding codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # inputs shape: (batch, embedding_dim, height, width)
        batch_size, _, height, width = inputs.shape
        
        # Flatten spatial dimensions: (batch, embedding_dim, height, width) -> (batch*height*width, embedding_dim)
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = flat_input.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        # (batch*height*width, embedding_dim) vs (num_embeddings, embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize
        quantized = self.embedding(encoding_indices)
        
        # Reshape back to original
        quantized = quantized.view(batch_size, height, width, self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, commitment_loss, encoding_indices


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 28x28 -> 14x14
        x = self.relu(self.conv1(x))
        # 14x14 -> 7x7
        x = self.relu(self.conv2(x))
        # 7x7 -> 7x7
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_channels):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_transpose1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 7x7 -> 7x7
        x = self.relu(self.conv1(x))
        # 7x7 -> 14x14
        x = self.relu(self.conv_transpose1(x))
        # 14x14 -> 28x28
        x = self.conv_transpose2(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, num_embeddings=64, 
                 embedding_dim=16, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, hidden_dim, embedding_dim)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dim, in_channels)
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Vector quantization
        quantized, vq_loss, encoding_indices = self.vector_quantizer(z)
        
        # Decode
        reconstruction = self.decoder(quantized)
        
        return reconstruction, quantized, vq_loss, encoding_indices