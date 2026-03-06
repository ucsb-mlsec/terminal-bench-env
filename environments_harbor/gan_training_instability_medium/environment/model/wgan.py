#!/usr/bin/env python3

import torch
import torch.nn as nn


class WGAN:
    """Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation."""
    
    def __init__(self, discriminator, generator, device):
        """
        Initialize WGAN-GP trainer.
        
        Args:
            discriminator: Discriminator/Critic network
            generator: Generator network
            device: Device to run computations on (cuda/cpu)
        """
        self.discriminator = discriminator
        self.generator = generator
        self.device = device
    
    def _gradient_penalty(self, real_samples, fake_samples):
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated fake samples
            
        Returns:
            Scalar gradient penalty value
        """
        # TODO: Implement gradient penalty computation
        batch_size = real_samples.size(0)
        
        # Sample random interpolation coefficients
        alpha = torch.rand(batch_size, 1, device=self.device)
        
        # Expand alpha to match sample dimensions
        for _ in range(len(real_samples.shape) - 2):
            alpha = alpha.unsqueeze(-1)
        
        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Compute discriminator output for interpolated samples
        d_interpolates = self.discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_samples, lambda_gp=10.0):
        """
        Perform one training step for the discriminator.
        
        Args:
            real_samples: Batch of real samples
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = real_samples.size(0)
        
        # Generate fake samples
        noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_samples = self.generator(noise).detach()
        
        # Discriminator outputs
        real_validity = self.discriminator(real_samples)
        fake_validity = self.discriminator(fake_samples)
        
        # Compute gradient penalty
        gp = self._gradient_penalty(real_samples, fake_samples)
        
        # Wasserstein loss with gradient penalty
        d_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gp
        
        return {
            'discriminator_loss': d_loss.item(),
            'gradient_penalty': gp.item(),
            'wasserstein_distance': (real_validity.mean() - fake_validity.mean()).item()
        }