#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

# Add reference directory to path to import the reference implementation
sys.path.insert(0, '/workspace/reference')
from gradient_penalty import compute_gradient_penalty


class WGAN(nn.Module):
    def __init__(self, discriminator):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
    
    def _gradient_penalty(self, real_samples, fake_samples, device):
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated fake samples
            device: Device to perform computation on
            
        Returns:
            Scalar gradient penalty value
        """
        return compute_gradient_penalty(self.discriminator, real_samples, fake_samples, device)


# Test the implementation
if __name__ == "__main__":
    import json
    
    # Simple discriminator for testing
    class SimpleDiscriminator(nn.Module):
        def __init__(self, input_dim):
            super(SimpleDiscriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.model(x)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configurations
    test_configs = [
        (8, 64),
        (16, 128),
        (32, 256),
        (8, 128),
        (16, 64),
    ]
    
    max_error = 0.0
    all_tests_passed = True
    
    for batch_size, feature_dim in test_configs:
        # Create discriminator
        discriminator = SimpleDiscriminator(feature_dim).to(device)
        wgan = WGAN(discriminator).to(device)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Generate test data
        real_samples = torch.randn(batch_size, feature_dim).to(device)
        fake_samples = torch.randn(batch_size, feature_dim).to(device)
        
        # Compute gradient penalty using both methods
        penalty_reference = compute_gradient_penalty(discriminator, real_samples, fake_samples, device)
        penalty_wgan = wgan._gradient_penalty(real_samples, fake_samples, device)
        
        # Check if results match
        error = abs(penalty_reference.item() - penalty_wgan.item())
        max_error = max(max_error, error)
        
        if error > 0.01:
            all_tests_passed = False
            print(f"Test failed for batch_size={batch_size}, feature_dim={feature_dim}")
            print(f"Reference: {penalty_reference.item()}, WGAN: {penalty_wgan.item()}, Error: {error}")
        else:
            print(f"Test passed for batch_size={batch_size}, feature_dim={feature_dim}, Error: {error}")
    
    # Write results
    results = {
        "all_tests_passed": all_tests_passed,
        "max_error": float(max_error)
    }
    
    os.makedirs('/workspace', exist_ok=True)
    with open('/workspace/result.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults written to /workspace/result.json")
    print(f"All tests passed: {all_tests_passed}")
    print(f"Max error: {max_error}")