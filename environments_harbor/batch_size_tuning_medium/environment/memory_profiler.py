#!/usr/bin/env python3

import sys
from typing import Dict, Tuple

def bytes_to_mb(bytes_value: int) -> float:
    """
    Convert bytes to megabytes.
    
    Args:
        bytes_value: Value in bytes
        
    Returns:
        Value in megabytes
    """
    return bytes_value / (1024 * 1024)


def estimate_memory_usage(
    num_parameters: int,
    batch_size: int,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: int = 4,
    optimizer_state_multiplier: int = 2
) -> Dict[str, float]:
    """
    Estimate memory usage for neural network training.
    
    This function calculates the memory requirements for training a neural network,
    including model parameters, gradients, optimizer state, and activations.
    
    Memory Components:
    1. Model Parameters: Storage for all trainable weights
    2. Gradients: Storage for gradients of all parameters (same size as parameters)
    3. Optimizer State: Additional memory for optimizer (e.g., Adam stores momentum and variance)
    4. Activations: Intermediate outputs stored during forward pass for backward pass
    
    Args:
        num_parameters: Total number of trainable parameters in the model
        batch_size: Number of samples processed in one training iteration
        input_shape: Tuple of (channels, height, width) for input images
        precision: Number of bytes per parameter (4 for FP32, 2 for FP16)
        optimizer_state_multiplier: Multiplier for optimizer memory overhead
                                   (2 for Adam which stores first and second moments)
    
    Returns:
        Dictionary containing memory estimates in MB:
        - model_mb: Memory for model parameters
        - gradients_mb: Memory for parameter gradients
        - optimizer_mb: Memory for optimizer state
        - activations_mb: Memory for layer activations
        - total_mb: Total estimated memory usage
    """
    # Calculate model parameters memory
    model_memory_bytes = num_parameters * precision
    model_mb = bytes_to_mb(model_memory_bytes)
    
    # Calculate gradients memory (same as model parameters)
    gradients_memory_bytes = num_parameters * precision
    gradients_mb = bytes_to_mb(gradients_memory_bytes)
    
    # Calculate optimizer state memory (Adam needs 2x parameter memory)
    optimizer_memory_bytes = num_parameters * precision * optimizer_state_multiplier
    optimizer_mb = bytes_to_mb(optimizer_memory_bytes)
    
    # Estimate activations memory
    # Activations depend on batch size and architecture depth
    # For typical CNNs, activations can be estimated using:
    # - Input size grows through layers but reduces with pooling
    # - Deep networks (ResNet-style) have many layers with residual connections
    # - Rule of thumb: activations ≈ batch_size * input_elements * depth_factor * precision
    
    channels, height, width = input_shape
    input_elements = channels * height * width
    
    # Estimate depth factor based on typical architectures
    # For ResNet-like models with ~50 layers, activation memory is roughly:
    # batch_size * input_elements * 15-20 (accounting for feature map expansion and multiple layers)
    # This is a heuristic that produces realistic memory estimates
    depth_factor = 18
    
    activations_memory_bytes = batch_size * input_elements * depth_factor * precision
    activations_mb = bytes_to_mb(activations_memory_bytes)
    
    # Calculate total memory
    total_mb = model_mb + gradients_mb + optimizer_mb + activations_mb
    
    return {
        'model_mb': model_mb,
        'gradients_mb': gradients_mb,
        'optimizer_mb': optimizer_mb,
        'activations_mb': activations_mb,
        'total_mb': total_mb
    }


def calculate_memory(
    num_parameters: int,
    batch_size: int,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: int = 4,
    optimizer_state_multiplier: int = 2
) -> Dict[str, float]:
    """
    Alias for estimate_memory_usage for backward compatibility.
    
    Args:
        num_parameters: Total number of trainable parameters in the model
        batch_size: Number of samples processed in one training iteration
        input_shape: Tuple of (channels, height, width) for input images
        precision: Number of bytes per parameter (4 for FP32, 2 for FP16)
        optimizer_state_multiplier: Multiplier for optimizer memory overhead
    
    Returns:
        Dictionary containing memory estimates in MB
    """
    return estimate_memory_usage(
        num_parameters=num_parameters,
        batch_size=batch_size,
        input_shape=input_shape,
        precision=precision,
        optimizer_state_multiplier=optimizer_state_multiplier
    )


def find_optimal_batch_size(
    num_parameters: int,
    memory_budget_mb: float,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    min_batch_size: int = 1,
    max_batch_size: int = 256,
    precision: int = 4,
    optimizer_state_multiplier: int = 2,
    safety_margin: float = 0.95
) -> Tuple[int, float]:
    """
    Find the largest power-of-2 batch size that fits within memory budget.
    
    Args:
        num_parameters: Total number of model parameters
        memory_budget_mb: Available memory in megabytes
        input_shape: Input dimensions
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size
        precision: Bytes per parameter
        optimizer_state_multiplier: Optimizer memory multiplier
        safety_margin: Use only this fraction of available memory (0.95 = 95%)
    
    Returns:
        Tuple of (optimal_batch_size, estimated_memory_mb)
    """
    effective_budget = memory_budget_mb * safety_margin
    
    # Generate power-of-2 batch sizes
    batch_sizes = []
    bs = 1
    while bs <= max_batch_size:
        if bs >= min_batch_size:
            batch_sizes.append(bs)
        bs *= 2
    
    # Find largest batch size that fits
    optimal_batch_size = 1
    optimal_memory = 0
    
    for bs in batch_sizes:
        memory_estimate = estimate_memory_usage(
            num_parameters=num_parameters,
            batch_size=bs,
            input_shape=input_shape,
            precision=precision,
            optimizer_state_multiplier=optimizer_state_multiplier
        )
        
        if memory_estimate['total_mb'] <= effective_budget:
            optimal_batch_size = bs
            optimal_memory = memory_estimate['total_mb']
        else:
            break
    
    return optimal_batch_size, optimal_memory


if __name__ == '__main__':
    # Example usage
    num_params = 25000000  # 25M parameters (typical for ResNet-50)
    batch_size = 32
    
    memory_stats = estimate_memory_usage(
        num_parameters=num_params,
        batch_size=batch_size
    )
    
    print(f"Memory Estimation for batch_size={batch_size}:")
    print(f"  Model Parameters: {memory_stats['model_mb']:.2f} MB")
    print(f"  Gradients: {memory_stats['gradients_mb']:.2f} MB")
    print(f"  Optimizer State: {memory_stats['optimizer_mb']:.2f} MB")
    print(f"  Activations: {memory_stats['activations_mb']:.2f} MB")
    print(f"  Total: {memory_stats['total_mb']:.2f} MB")