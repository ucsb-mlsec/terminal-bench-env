#!/usr/bin/env python3
"""
Cross-entropy loss function implementation for multi-class classification.
This module provides loss computation for neural network training.
"""

import numpy as np


def targets_to_onehot(targets, num_classes=10):
    """
    Convert integer class labels to one-hot encoded vectors.
    
    Args:
        targets: numpy array of shape (batch_size,) with integer class labels
        num_classes: number of classes (default: 10)
    
    Returns:
        numpy array of shape (batch_size, num_classes) with one-hot vectors
    """
    batch_size = targets.shape[0]
    onehot = np.zeros((batch_size, num_classes), dtype=np.float32)
    onehot[np.arange(batch_size), targets] = 1.0
    return onehot


def cross_entropy_loss(predictions, targets, smoothing=0.0):
    """
    Compute cross-entropy loss for multi-class classification.
    
    This function computes the cross-entropy loss between predicted probabilities
    and target class labels. Supports label smoothing to reduce overconfidence.
    
    Args:
        predictions: numpy array of shape (batch_size, num_classes) containing
                    predicted probabilities for each class (should sum to 1)
        targets: numpy array of shape (batch_size,) containing integer class
                labels in range [0, num_classes-1]
        smoothing: label smoothing factor (float between 0.0 and 1.0)
                  0.0 = no smoothing (hard targets)
                  > 0.0 = smooth targets by redistributing probability mass
    
    Returns:
        scalar loss value (float) - mean cross-entropy loss across the batch
    
    Example:
        >>> predictions = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        >>> targets = np.array([1, 0])
        >>> loss = cross_entropy_loss(predictions, targets)
    """
    # Get number of classes from predictions shape
    num_classes = predictions.shape[1]
    batch_size = predictions.shape[0]
    
    # Convert targets to one-hot encoding
    onehot_targets = targets_to_onehot(targets, num_classes)
    
    # Apply label smoothing if specified
    if smoothing > 0.0:
        # Distribute smoothing mass uniformly across all classes
        # For correct class: probability = (1 - smoothing) + smoothing/num_classes
        # For incorrect classes: probability = smoothing/num_classes
        smoothing_value = smoothing / num_classes
        onehot_targets = onehot_targets * (1.0 - smoothing) + smoothing_value
    
    # Add small epsilon to predictions for numerical stability (avoid log(0))
    epsilon = 1e-15
    predictions_clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
    
    # Compute cross-entropy: -sum(target * log(prediction))
    log_predictions = np.log(predictions_clipped)
    sample_losses = -np.sum(onehot_targets * log_predictions, axis=1)
    
    # Return mean loss across batch
    mean_loss = np.mean(sample_losses)
    
    return mean_loss