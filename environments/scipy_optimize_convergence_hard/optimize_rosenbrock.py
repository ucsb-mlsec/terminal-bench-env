#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    """
    Rosenbrock function: f(x) = (a-x[0])^2 + b*(x[1]-x[0]^2)^2
    Global minimum at (1, 1) with f(1, 1) = 0
    """
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def main():
    print("=" * 60)
    print("Rosenbrock Function Optimization")
    print("=" * 60)
    print("\nTheoretical minimum: x = [1.0, 1.0], f(x) = 0.0")
    print("-" * 60)
    
    # Starting point far from the optimum
    x0 = np.array([0.0, 0.0])
    print(f"\nStarting point: x0 = {x0}")
    print(f"Initial function value: f(x0) = {rosenbrock(x0):.6f}")
    
    # Perform optimization with poor settings
    # Issues:
    # 1. Using Nelder-Mead method which is derivative-free and slow
    # 2. Default/loose tolerances
    # 3. Limited maximum iterations
    result = minimize(
        rosenbrock,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 100,  # Too few iterations
            'xatol': 1e-2,   # Loose tolerance on x
            'fatol': 1e-2    # Loose tolerance on function value
        }
    )
    
    print("\n" + "-" * 60)
    print("Optimization Results:")
    print("-" * 60)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"\nFinal point: x = [{result.x[0]:.4f}, {result.x[1]:.4f}]")
    print(f"Final function value: f(x) = {result.fun:.6f}")
    print(f"\nError from true minimum: {np.linalg.norm(result.x - np.array([1.0, 1.0])):.6f}")
    print("=" * 60)

if __name__ == "__main__":
    main()