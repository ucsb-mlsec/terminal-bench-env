#!/usr/bin/env python3
"""
Data transformation functions for the processing framework.
This module provides utilities for scaling, transforming, and standardizing data.
"""

from scicore import math_utils


def scale_data(data, factor=1.0):
    """
    Multiply all values in the data by a scaling factor.
    
    Args:
        data: A list or iterable of numeric values to scale
        factor: The scaling factor to apply (default: 1.0)
    
    Returns:
        A list of scaled values
    """
    return [x * factor for x in data]


def apply_function(data, func):
    """
    Apply a function to each element in the data.
    
    Args:
        data: A list or iterable of values to transform
        func: A function to apply to each element
    
    Returns:
        A list of transformed values
    """
    return [func(x) for x in data]


def standardize(data):
    """
    Standardize data to zero mean and unit variance.
    
    Uses scicore.math_utils functions to compute mean and variance,
    then transforms the data to have mean=0 and variance=1.
    
    Args:
        data: A list or iterable of numeric values to standardize
    
    Returns:
        A list of standardized values
    """
    mean = math_utils.compute_mean(data)
    variance = math_utils.compute_variance(data)
    
    # Handle edge case of zero variance
    if variance == 0:
        return [0.0 for _ in data]
    
    std_dev = variance ** 0.5
    return [(x - mean) / std_dev for x in data]