#!/usr/bin/env python3

import numpy as np
import json
from scipy.optimize import differential_evolution, minimize
import sys
sys.path.insert(0, '/workspace')
from objective_function import objective_function

# Set random seed for reproducibility
np.random.seed(42)

# Track function evaluations
evaluation_count = 0
best_score = float('inf')
best_params = None

def wrapped_objective(x):
    """Wrapper to track evaluations and best results"""
    global evaluation_count, best_score, best_params
    
    # Extract parameters
    x1, x2, x3, x4 = x[0], x[1], int(round(x[2])), int(round(x[3]))
    
    # Ensure integer constraints
    x3 = max(1, min(20, x3))
    x4 = max(1, min(20, x4))
    
    # Evaluate objective function
    score = objective_function(x1, x2, x3, x4)
    evaluation_count += 1
    
    # Track best result
    if score < best_score:
        best_score = score
        best_params = {'x1': float(x1), 'x2': float(x2), 'x3': x3, 'x4': x4}
        print(f"Evaluation {evaluation_count}: New best score = {best_score:.4f}")
        print(f"  Parameters: x1={x1:.4f}, x2={x2:.4f}, x3={x3}, x4={x4}")
    
    return score

# Define bounds for optimization
bounds = [
    (-5.0, 5.0),  # x1
    (-5.0, 5.0),  # x2
    (1, 20),      # x3
    (1, 20)       # x4
]

print("Starting optimization with Differential Evolution...")
print("=" * 60)

# Use Differential Evolution for global optimization
# This algorithm is well-suited for expensive black-box functions
result = differential_evolution(
    wrapped_objective,
    bounds=bounds,
    seed=42,
    maxiter=30,  # Limit iterations to control total evaluations
    popsize=4,   # Small population to stay within budget
    atol=0.01,
    tol=0.01,
    workers=1,
    updating='immediate',
    strategy='best1bin'
)

print("\n" + "=" * 60)
print(f"Differential Evolution completed")
print(f"Total evaluations: {evaluation_count}")
print(f"Best score found: {best_score:.4f}")
print(f"Best parameters: {best_params}")

# If we haven't reached minimum evaluations, do additional sampling
if evaluation_count < 100:
    print(f"\nPerforming additional evaluations to reach minimum of 100...")
    
    # Local refinement around best solution
    remaining = 100 - evaluation_count
    
    # Try variations around the best parameters found
    for i in range(remaining):
        # Add small random perturbations
        noise_scale = max(0.5, 2.0 * (1 - i / remaining))  # Decrease noise over time
        
        x1_test = best_params['x1'] + np.random.uniform(-noise_scale, noise_scale)
        x2_test = best_params['x2'] + np.random.uniform(-noise_scale, noise_scale)
        x3_test = best_params['x3'] + np.random.randint(-2, 3)
        x4_test = best_params['x4'] + np.random.randint(-2, 3)
        
        # Clip to bounds
        x1_test = np.clip(x1_test, -5.0, 5.0)
        x2_test = np.clip(x2_test, -5.0, 5.0)
        x3_test = int(np.clip(x3_test, 1, 20))
        x4_test = int(np.clip(x4_test, 1, 20))
        
        score = objective_function(x1_test, x2_test, x3_test, x4_test)
        evaluation_count += 1
        
        if score < best_score:
            best_score = score
            best_params = {'x1': float(x1_test), 'x2': float(x2_test), 'x3': x3_test, 'x4': x4_test}
            print(f"Evaluation {evaluation_count}: New best score = {best_score:.4f}")
            print(f"  Parameters: x1={x1_test:.4f}, x2={x2_test:.4f}, x3={x3_test}, x4={x4_test}")

print("\n" + "=" * 60)
print("Optimization complete!")
print(f"Total function evaluations: {evaluation_count}")
print(f"Final best score: {best_score:.4f}")
print(f"Final best parameters: {best_params}")

# Verify the result
verification_score = objective_function(
    best_params['x1'],
    best_params['x2'],
    best_params['x3'],
    best_params['x4']
)
print(f"\nVerification score: {verification_score:.4f}")

if verification_score <= 1.80:
    print("✓ Target score of 1.80 or lower achieved!")
else:
    print(f"✗ Target not met. Score is {verification_score:.4f}, need ≤ 1.80")

# Save best parameters to JSON file
output_path = '/workspace/best_params.json'
with open(output_path, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"\nBest parameters saved to {output_path}")