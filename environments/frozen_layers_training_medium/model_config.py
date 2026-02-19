#!/usr/bin/env python3

import torch
import torch.nn as nn
import json

# Load the model from model_config.py
import sys
sys.path.insert(0, '/workspace')
from model_config import MyModel

def count_parameters(model, requires_grad=None):
    """
    Count parameters in a model.
    
    Args:
        model: PyTorch model
        requires_grad: If None, count all parameters. If True/False, count only parameters with that requires_grad value.
    
    Returns:
        Total number of parameters
    """
    if requires_grad is None:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)

def freeze_all_except_final(model):
    """
    Freeze all parameters except the final layer.
    Assumes the final layer is named 'final_layer' or is the last Linear layer.
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Find the final layer - try common naming conventions
    final_layer = None
    if hasattr(model, 'final_layer'):
        final_layer = model.final_layer
    elif hasattr(model, 'output_layer'):
        final_layer = model.output_layer
    elif hasattr(model, 'classifier'):
        final_layer = model.classifier
    elif hasattr(model, 'fc'):
        final_layer = model.fc
    else:
        # Find the last Linear layer by iterating through all modules
        linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if linear_layers:
            final_layer = linear_layers[-1]
    
    # Unfreeze the final layer
    if final_layer is not None:
        for param in final_layer.parameters():
            param.requires_grad = True
    
    return model

def main():
    # Create model instance
    model = MyModel()
    
    # Calculate total trainable parameters (all layers trainable)
    total_params = count_parameters(model, requires_grad=True)
    
    # Freeze all layers except the final one
    model = freeze_all_except_final(model)
    
    # Calculate trainable parameters after freezing
    frozen_params = count_parameters(model, requires_grad=True)
    
    # Calculate reduction percentage
    if total_params > 0:
        reduction_percent = ((total_params - frozen_params) / total_params) * 100
        reduction_percent = round(reduction_percent, 2)
    else:
        reduction_percent = 0.0
    
    # Prepare results
    results = {
        "total_params": total_params,
        "frozen_params": frozen_params,
        "reduction_percent": reduction_percent
    }
    
    # Save to JSON file
    with open('/workspace/analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete!")
    print(f"Total parameters (all trainable): {total_params:,}")
    print(f"Trainable parameters (only final layer): {frozen_params:,}")
    print(f"Reduction: {reduction_percent}%")

if __name__ == "__main__":
    main()