#!/usr/bin/env python3

import torch


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP.

    This is the reference implementation that computes the gradient penalty
    as described in "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017).

    Args:
        discriminator: The discriminator/critic network
        real_samples: Real data samples tensor
        fake_samples: Generated fake samples tensor
        device: Device to perform computation on

    Returns:
        Scalar gradient penalty value
    """
    batch_size = real_samples.size(0)

    # Sample random interpolation coefficients
    alpha = torch.rand(batch_size, 1, device=device)

    # Expand alpha to match sample dimensions
    for _ in range(len(real_samples.shape) - 2):
        alpha = alpha.unsqueeze(-1)

    # Interpolate between real and fake samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    # Compute discriminator output for interpolated samples
    d_interpolates = discriminator(interpolates)

    # Compute gradients of discriminator output w.r.t. interpolated samples
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

    # Compute gradient penalty: E[(||grad|| - 1)^2]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty
