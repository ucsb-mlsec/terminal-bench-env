#!/usr/bin/env python3

import torch
import torch.nn as nn
import json
import sys
import os

# Add workspace to path to enable imports
sys.path.insert(0, '/workspace')

from reference.gradient_penalty import compute_gradient_penalty
from model.wgan import WGAN


class SimpleDiscriminator(nn.Module):
    """Simple discriminator for testing gradient penalty"""
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


def test_gradient_penalty():
    """Test gradient penalty implementation against reference"""
    
    device = 'cpu'
    test_cases = []
    batch_sizes = [8, 16, 32]
    dimensions = [64, 128, 256]
    
    max_error = 0.0
    all_passed = True
    tolerance = 0.01
    
    print("Starting gradient penalty tests...")
    print("=" * 60)
    
    test_id = 0
    for batch_size in batch_sizes:
        for dim in dimensions:
            test_id += 1
            
            # Set random seed for reproducibility
            seed = 42 + test_id
            torch.manual_seed(seed)
            
            # Create test data
            real_samples = torch.randn(batch_size, dim, device=device)
            fake_samples = torch.randn(batch_size, dim, device=device)
            
            # Create discriminator
            discriminator = SimpleDiscriminator(dim).to(device)
            
            # Initialize discriminator weights with a fixed seed
            torch.manual_seed(seed)
            for param in discriminator.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
            
            # Compute reference gradient penalty
            reference_gp = compute_gradient_penalty(
                discriminator, real_samples, fake_samples, device
            )
            
            # Create WGAN instance and compute gradient penalty
            wgan = WGAN(discriminator)
            implementation_gp = wgan._gradient_penalty(
                real_samples, fake_samples, device
            )
            
            # Compute error
            error = abs(reference_gp - implementation_gp)
            max_error = max(max_error, error)
            
            # Check if test passed
            passed = error < tolerance
            if not passed:
                all_passed = False
            
            # Print test result
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"Test {test_id}: batch_size={batch_size:2d}, dim={dim:3d} | "
                  f"Reference={reference_gp:.6f}, "
                  f"Implementation={implementation_gp:.6f}, "
                  f"Error={error:.6f} | {status}")
            
            test_cases.append({
                'batch_size': batch_size,
                'dim': dim,
                'reference': float(reference_gp),
                'implementation': float(implementation_gp),
                'error': float(error),
                'passed': passed
            })
    
    print("=" * 60)
    print(f"\nTotal tests: {len(test_cases)}")
    print(f"Passed: {sum(1 for tc in test_cases if tc['passed'])}")
    print(f"Failed: {sum(1 for tc in test_cases if not tc['passed'])}")
    print(f"Maximum error: {max_error:.6f}")
    print(f"Tolerance: {tolerance}")
    print(f"\nAll tests passed: {all_passed}")
    
    # Prepare results
    results = {
        'all_tests_passed': all_passed,
        'max_error': float(max_error)
    }
    
    # Write results to JSON file
    os.makedirs('/workspace', exist_ok=True)
    with open('/workspace/result.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults written to /workspace/result.json")
    
    return all_passed


if __name__ == '__main__':
    success = test_gradient_penalty()
    sys.exit(0 if success else 1)