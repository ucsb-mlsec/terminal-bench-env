#!/usr/bin/env python3

def compute_gradient_flow(block_outputs, depth):
    """
    Compute gradient flow through a residual block.
    
    Args:
        block_outputs: List of [input_activation, residual_path_output, skip_connection_output]
        depth: Layer depth (1 to 100)
    
    Returns:
        Gradient magnitude at this depth
    """
    input_activation, residual_path_output, skip_connection_output = block_outputs
    
    # Compute gradient through residual path
    residual_gradient = residual_path_output * 0.9
    
    # BUG: Gradient is scaled by depth, causing it to explode at deep layers
    gradient_magnitude = (residual_gradient + skip_connection_output) * depth
    
    return gradient_magnitude


# Test data
if __name__ == '__main__':
    test_inputs = [1.0, 0.8, 1.0]
    test_depths = [1, 10, 25, 50, 75, 100]
    
    print("Testing gradient flow at various depths:")
    print("-" * 50)
    for depth in test_depths:
        gradient = compute_gradient_flow(test_inputs, depth)
        status = "✓ STABLE" if 0.1 <= gradient <= 10.0 else "✗ UNSTABLE"
        print(f"Depth {depth:3d}: Gradient = {gradient:8.2f} {status}")
    
    print("\nNote: Gradients should remain between 0.1 and 10.0 at all depths")