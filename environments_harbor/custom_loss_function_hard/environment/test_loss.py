#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss for quality tier ranking.
    
    This loss function penalizes predictions proportionally to their distance
    from the true quality tier. It uses a distance-weighted cross-entropy approach
    where the penalty increases with the absolute difference between predicted
    and true tiers.
    
    The loss is computed as:
    1. Convert logits to probabilities via softmax
    2. For each class, compute distance weight: |predicted_class - true_class|
    3. Weight the cross-entropy loss by these distances
    
    This ensures predictions closer to the true tier incur lower penalties while
    maintaining differentiability for gradient-based optimization.
    """
    
    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()
    
    def forward(self, predictions, targets):
        """
        Compute ordinal regression loss.
        
        Args:
            predictions: Tensor of shape (batch_size, num_classes) containing logits
            targets: Tensor of shape (batch_size,) containing class indices (0 to num_classes-1)
        
        Returns:
            Scalar loss tensor
        """
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        
        # Convert logits to log probabilities for numerical stability
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Create distance weight matrix
        # For each true class, compute distance to all possible classes
        class_indices = torch.arange(num_classes, device=predictions.device, dtype=torch.float32)
        targets_expanded = targets.unsqueeze(1).float()  # (batch_size, 1)
        
        # Distance weights: |predicted_class - true_class|
        distance_weights = torch.abs(class_indices.unsqueeze(0) - targets_expanded)  # (batch_size, num_classes)
        
        # Add 1 to ensure correct predictions still have some loss (standard cross-entropy behavior)
        # This prevents zero loss and maintains gradient flow
        distance_weights = distance_weights + 1.0
        
        # Compute weighted negative log likelihood
        # Standard NLL: -log_probs[i, targets[i]]
        # Weighted NLL: sum over all classes of (distance_weight * prob * log_prob equivalent)
        
        # Convert to probabilities for weighting
        probs = torch.exp(log_probs)
        
        # Compute expected distance: sum of (distance * probability) for each class
        weighted_distances = (distance_weights * probs).sum(dim=1)
        
        # Alternative approach: Use distance-weighted NLL
        # The loss should be higher when probability mass is on classes far from target
        loss = (distance_weights * probs).sum(dim=1).mean()
        
        return loss


def main():
    """Test the OrdinalRegressionLoss implementation."""
    
    print("Testing OrdinalRegressionLoss...")
    
    # Initialize criterion
    criterion = OrdinalRegressionLoss()
    
    all_tests_passed = True
    
    # Test 1: Loss ordering
    print("\nTest 1: Loss ordering")
    try:
        target = torch.tensor([3])
        
        # Create logits that strongly predict class 2 (one tier off)
        logits_class2 = torch.tensor([[−2.0, -1.0, 3.0, -1.0, -2.0]], requires_grad=True)
        
        # Create logits that strongly predict class 0 (three tiers off)
        logits_class0 = torch.tensor([[3.0, -1.0, -2.0, -2.0, -3.0]], requires_grad=True)
        
        loss_class2 = criterion(logits_class2, target)
        loss_class0 = criterion(logits_class0, target)
        
        print(f"  Loss for pred=class2 (1 tier off): {loss_class2.item():.6f}")
        print(f"  Loss for pred=class0 (3 tiers off): {loss_class0.item():.6f}")
        
        if loss_class2.item() < loss_class0.item():
            print("  ✓ Loss ordering test PASSED")
        else:
            print("  ✗ Loss ordering test FAILED")
            all_tests_passed = False
    except Exception as e:
        print(f"  ✗ Loss ordering test FAILED with exception: {e}")
        all_tests_passed = False
    
    # Test 2: Gradient computation
    print("\nTest 2: Gradient computation")
    try:
        predictions = torch.tensor([[1.0, 2.0, -1.0, -1.0, -2.0]], requires_grad=True)
        targets = torch.tensor([1])
        
        loss = criterion(predictions, targets)
        loss.backward()
        
        if predictions.grad is not None and torch.any(predictions.grad != 0):
            print(f"  Gradients computed: {predictions.grad}")
            print("  ✓ Gradient computation test PASSED")
        else:
            print("  ✗ Gradient computation test FAILED: No gradients or all zero")
            all_tests_passed = False
    except Exception as e:
        print(f"  ✗ Gradient computation test FAILED with exception: {e}")
        all_tests_passed = False
    
    # Test 3: Batch consistency
    print("\nTest 3: Batch consistency")
    try:
        # Create batch of predictions
        batch_predictions = torch.tensor([
            [2.0, 1.0, -1.0, -2.0, -3.0],
            [-1.0, 2.0, 1.0, -1.0, -2.0],
            [-2.0, -1.0, 2.0, 1.0, -1.0],
            [-3.0, -2.0, -1.0, 2.0, 1.0],
        ], requires_grad=True)
        batch_targets = torch.tensor([0, 1, 2, 3])
        
        # Compute batch loss
        batch_loss = criterion(batch_predictions, batch_targets)
        
        # Compute individual losses
        individual_losses = []
        for i in range(batch_predictions.size(0)):
            pred = batch_predictions[i:i+1]
            tgt = batch_targets[i:i+1]
            individual_losses.append(criterion(pred, tgt).item())
        
        mean_individual = sum(individual_losses) / len(individual_losses)
        
        print(f"  Batch loss: {batch_loss.item():.6f}")
        print(f"  Mean of individual losses: {mean_individual:.6f}")
        
        if torch.allclose(batch_loss, torch.tensor(mean_individual), rtol=1e-4, atol=1e-6):
            print("  ✓ Batch consistency test PASSED")
        else:
            print("  ✗ Batch consistency test FAILED")
            all_tests_passed = False
    except Exception as e:
        print(f"  ✗ Batch consistency test FAILED with exception: {e}")
        all_tests_passed = False
    
    # Test 4: Numerical stability
    print("\nTest 4: Numerical stability")
    try:
        # Test with extreme logit values
        extreme_predictions = torch.tensor([
            [1e10, -1e10, 0.0, -1e10, -1e10],
            [-1e10, 1e10, -1e10, 0.0, -1e10],
            [1e5, -1e5, 1e5, -1e5, 1e5],
        ], requires_grad=True)
        extreme_targets = torch.tensor([0, 1, 2])
        
        loss = criterion(extreme_predictions, extreme_targets)
        loss.backward()
        
        if torch.isfinite(loss) and torch.all(torch.isfinite(extreme_predictions.grad)):
            print(f"  Loss with extreme values: {loss.item():.6f}")
            print("  ✓ Numerical stability test PASSED")
        else:
            print("  ✗ Numerical stability test FAILED: NaN or Inf detected")
            all_tests_passed = False
    except Exception as e:
        print(f"  ✗ Numerical stability test FAILED with exception: {e}")
        all_tests_passed = False
    
    # Write results
    import os
    os.makedirs('/workspace/ranking_system', exist_ok=True)
    
    with open('/workspace/ranking_system/results.txt', 'w') as f:
        if all_tests_passed:
            f.write('STATUS=PASS\n')
            print("\n" + "="*50)
            print("ALL TESTS PASSED")
            print("="*50)
        else:
            f.write('STATUS=FAIL\n')
            print("\n" + "="*50)
            print("SOME TESTS FAILED")
            print("="*50)


if __name__ == '__main__':
    main()