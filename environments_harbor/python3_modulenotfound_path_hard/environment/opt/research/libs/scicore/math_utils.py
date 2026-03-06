#!/usr/bin/env python3
"""
Mathematical utility functions for scientific computing.
"""

def compute_mean(data):
    """
    Compute the arithmetic mean of a list of numbers.
    
    Args:
        data: List of numeric values
        
    Returns:
        float: The mean of the data
    """
    if not data:
        return 0
    return sum(data) / len(data)


def compute_variance(data):
    """
    Compute the variance of a list of numbers.
    
    Args:
        data: List of numeric values
        
    Returns:
        float: The variance of the data
    """
    if not data:
        return 0
    mean = compute_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)


def normalize_vector(vec):
    """
    Normalize a vector to unit length.
    
    Args:
        vec: List of numeric values representing a vector
        
    Returns:
        list: Normalized vector
    """
    if not vec:
        return []
    magnitude = sum(x ** 2 for x in vec) ** 0.5
    if magnitude == 0:
        return vec
    return [x / magnitude for x in vec]


def factorial(n):
    """
    Compute the factorial of a non-negative integer.
    
    Args:
        n: Non-negative integer
        
    Returns:
        int: n! (factorial of n)
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result