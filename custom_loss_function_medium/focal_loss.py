#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the true class
    
    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive/negative examples
        gamma (float): Focusing parameter for modulating loss. Higher gamma increases focus on hard examples
    """
    
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Forward pass of focal loss.
        
        Args:
            inputs (torch.Tensor): Logits from the model, shape (N, C) where N is batch size, C is number of classes
            targets (torch.Tensor): Ground truth class labels, shape (N,) with values in [0, C-1]
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Get probabilities from logits using softmax
        probs = F.softmax(inputs, dim=1)
        
        # Get the probability of the true class for each sample
        # Gather the probabilities corresponding to the target classes
        batch_size = inputs.size(0)
        p_t = probs[range(batch_size), targets]
        
        # Compute focal loss components
        # FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t)
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Return mean loss across the batch
        return focal_loss.mean()


# Validation script
def validate_focal_loss():
    """Validate the FocalLoss implementation"""
    
    results = {
        'output_valid': False,
        'gradients_valid': False,
        'weighting_valid': False,
        'overall_correct': False
    }
    
    try:
        # Initialize the focal loss
        focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Test 1: Output validity (non-negative scalar)
        print("Test 1: Checking output validity...")
        inputs = torch.randn(10, 5, requires_grad=True)
        targets = torch.randint(0, 5, (10,))
        
        loss = focal_loss_fn(inputs, targets)
        
        # Check if output is scalar
        is_scalar = loss.dim() == 0 or (loss.dim() == 1 and loss.size(0) == 1)
        # Check if output is non-negative
        is_non_negative = loss.item() >= 0
        
        results['output_valid'] = is_scalar and is_non_negative
        print(f"  Output is scalar: {is_scalar}")
        print(f"  Output is non-negative: {is_non_negative}")
        print(f"  Loss value: {loss.item()}")
        
        # Test 2: Gradient validity
        print("\nTest 2: Checking gradient validity...")
        try:
            loss.backward()
            has_gradients = inputs.grad is not None
            gradients_finite = torch.isfinite(inputs.grad).all().item()
            results['gradients_valid'] = has_gradients and gradients_finite
            print(f"  Gradients computed: {has_gradients}")
            print(f"  Gradients finite: {gradients_finite}")
        except Exception as e:
            print(f"  Gradient computation failed: {e}")
            results['gradients_valid'] = False
        
        # Test 3: Easy vs hard example weighting
        print("\nTest 3: Checking easy vs hard example weighting...")
        
        # Create easy example (confident correct prediction)
        easy_inputs = torch.tensor([[10.0, -5.0, -5.0, -5.0, -5.0]], requires_grad=True)
        easy_targets = torch.tensor([0])  # Target is class 0, which has high logit
        
        # Create hard example (uncertain/wrong prediction)
        hard_inputs = torch.tensor([[-1.0, 0.5, 0.3, 0.2, 0.1]], requires_grad=True)
        hard_targets = torch.tensor([0])  # Target is class 0, which has low logit
        
        easy_loss = focal_loss_fn(easy_inputs, easy_targets)
        hard_loss = focal_loss_fn(hard_inputs, hard_targets)
        
        print(f"  Easy example loss: {easy_loss.item()}")
        print(f"  Hard example loss: {hard_loss.item()}")
        print(f"  Hard loss > Easy loss: {hard_loss.item() > easy_loss.item()}")
        
        # Easy loss should be much smaller than hard loss
        results['weighting_valid'] = hard_loss.item() > easy_loss.item()
        
        # Additional test: Compare with standard cross-entropy
        print("\nAdditional test: Comparing with standard cross-entropy...")
        ce_loss_fn = nn.CrossEntropyLoss()
        
        # For easy example
        easy_ce = ce_loss_fn(easy_inputs, easy_targets)
        easy_focal = focal_loss_fn(easy_inputs, easy_targets)
        
        # For hard example
        hard_ce = ce_loss_fn(hard_inputs, hard_targets)
        hard_focal = focal_loss_fn(hard_inputs, hard_targets)
        
        print(f"  Easy - CE: {easy_ce.item():.6f}, Focal: {easy_focal.item():.6f}")
        print(f"  Hard - CE: {hard_ce.item():.6f}, Focal: {hard_focal.item():.6f}")
        
        # Focal loss should down-weight easy examples more
        easy_ratio = easy_focal.item() / easy_ce.item() if easy_ce.item() > 0 else 0
        hard_ratio = hard_focal.item() / hard_ce.item() if hard_ce.item() > 0 else 0
        
        print(f"  Easy Focal/CE ratio: {easy_ratio:.6f}")
        print(f"  Hard Focal/CE ratio: {hard_ratio:.6f}")
        print(f"  Focusing behavior (easy ratio < hard ratio): {easy_ratio < hard_ratio}")
        
        # Overall correctness
        results['overall_correct'] = all([
            results['output_valid'],
            results['gradients_valid'],
            results['weighting_valid']
        ])
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
    
    return results


if __name__ == "__main__":
    print("Validating Focal Loss Implementation...")
    print("=" * 60)
    
    results = validate_focal_loss()
    
    print("\n" + "=" * 60)
    print("Validation Results:")
    print("=" * 60)
    
    # Write results to file
    with open('/workspace/validation_result.txt', 'w') as f:
        f.write(f"output_valid: {'yes' if results['output_valid'] else 'no'}\n")
        f.write(f"gradients_valid: {'yes' if results['gradients_valid'] else 'no'}\n")
        f.write(f"weighting_valid: {'yes' if results['weighting_valid'] else 'no'}\n")
        f.write(f"overall_correct: {'yes' if results['overall_correct'] else 'no'}\n")
    
    print("Results written to /workspace/validation_result.txt")
    print("\nSummary:")
    for key, value in results.items():
        print(f"  {key}: {'yes' if value else 'no'}")