#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    """
    Rosenbrock function - standard optimization benchmark.
    Global minimum at x = [1, 1, ..., 1] with f(x) = 0
    """
    n = len(x)
    result = 0.0
    for i in range(n - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result

def test_optimization():
    """
    Test optimization for different dimensions.
    This version has bugs that prevent successful convergence.
    """
    dimensions = [2, 5, 10]
    
    for dim in dimensions:
        # Generate initial point following the pattern [-1.2, 1.0, -1.2, 1.0, ...]
        x0 = np.array([(-1.2 if i % 2 == 0 else 1.0) for i in range(dim)])
        
        # BUG 1: Using Nelder-Mead without sufficient iterations
        # BUG 2: Default maxiter for Nelder-Mead is too low for higher dimensions
        # BUG 3: Not using gradient information (inefficient)
        result = minimize(
            rosenbrock,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100}  # Way too few iterations!
        )
        
        final_value = result.fun
        success = result.success
        
        print(f"{dim},{final_value:.3e},{success}")

if __name__ == "__main__":
    test_optimization()