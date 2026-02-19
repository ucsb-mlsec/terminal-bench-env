#!/usr/bin/env python3

# Constants for learning rate scheduling
BASE_LR = 0.001
WARMUP_EPOCHS = 10
DECAY_START_EPOCH = 60
TOTAL_EPOCHS = 100

def get_learning_rate(epoch):
    """
    Calculate learning rate for given epoch with warmup and decay phases.
    
    Args:
        epoch: Current epoch number (0-indexed)
    
    Returns:
        Learning rate value for the given epoch
    """
    
    # Warmup phase: linearly increase from 0.0001 to BASE_LR over first 10 epochs
    if epoch < WARMUP_EPOCHS:
        # Linear warmup from 0.0001 to 0.001
        start_lr = 0.0001
        warmup_lr = start_lr + (BASE_LR - start_lr) * (epoch / WARMUP_EPOCHS)
        return warmup_lr
    
    # Steady phase: maintain BASE_LR from epoch 10 to epoch 60
    elif epoch < DECAY_START_EPOCH:
        return BASE_LR
    
    # Decay phase: exponential decay after epoch 60
    else:
        # Calculate decay factor
        # We want to reach less than 0.0005 by epoch 99
        # Using exponential decay: lr = BASE_LR * decay_rate^(epochs_since_decay_start)
        epochs_in_decay = epoch - DECAY_START_EPOCH
        total_decay_epochs = TOTAL_EPOCHS - DECAY_START_EPOCH
        
        # Using exponential decay with factor that reaches ~0.00025 at epoch 99
        decay_rate = 0.98
        decayed_lr = BASE_LR * (decay_rate ** epochs_in_decay)
        
        return decayed_lr