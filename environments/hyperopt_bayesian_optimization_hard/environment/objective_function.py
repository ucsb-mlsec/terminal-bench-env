#!/usr/bin/env python3

import numpy as np

def objective_function(alpha, beta, gamma, delta, epsilon, zeta):
    """
    Complex optimization function with multiple local minima.
    
    Parameters:
    - alpha: [0, 1]
    - beta: [0, 10]
    - gamma: [0, 1]
    - delta: [0, 5]
    - epsilon: [0, 1]
    - zeta: [0, 3]
    
    Returns:
    - float: objective score (lower is better)
    """
    
    # Optimal values (not documented for the user)
    # Global minimum around: alpha=0.3, beta=5.0, gamma=0.7, delta=2.5, epsilon=0.5, zeta=1.5
    
    # Main quadratic terms with different weights
    term1 = 10 * (alpha - 0.3)**2
    term2 = 0.5 * (beta - 5.0)**2
    term3 = 8 * (gamma - 0.7)**2
    term4 = 1.2 * (delta - 2.5)**2
    term5 = 6 * (epsilon - 0.5)**2
    term6 = 2 * (zeta - 1.5)**2
    
    # Trigonometric terms to create local minima
    trig1 = 2 * np.sin(3 * np.pi * alpha) * np.cos(2 * np.pi * beta / 10)
    trig2 = 1.5 * np.cos(4 * np.pi * gamma) * np.sin(np.pi * delta / 5)
    trig3 = 1.0 * np.sin(2 * np.pi * epsilon) * np.cos(np.pi * zeta / 3)
    
    # Cross-term interactions
    cross1 = 0.8 * (alpha - 0.3) * (beta - 5.0)
    cross2 = 0.5 * (gamma - 0.7) * (delta - 2.5)
    cross3 = 0.3 * (epsilon - 0.5) * (zeta - 1.5)
    
    # Additional complexity with exponential terms (small coefficients)
    exp1 = 0.5 * np.exp(-((alpha - 0.8)**2 + (gamma - 0.2)**2))
    exp2 = 0.3 * np.exp(-((beta - 2.0)**2 / 10 + (delta - 4.0)**2 / 5))
    
    # Combine all terms
    score = (term1 + term2 + term3 + term4 + term5 + term6 + 
             trig1 + trig2 + trig3 + 
             cross1 + cross2 + cross3 - 
             exp1 - exp2 + 
             5.0)  # Baseline offset
    
    return float(score)