#!/usr/bin/env python3
"""
Core scientific utilities library - data operations module.
Provides essential data manipulation functions for research computing.
"""


def filter_outliers(data, threshold=2.0):
    """
    Remove values beyond threshold standard deviations from the mean.
    
    Args:
        data: List of numeric values
        threshold: Number of standard deviations (default: 2.0)
    
    Returns:
        List of values within threshold standard deviations
    """
    if not data:
        return []
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    
    if std_dev == 0:
        return data
    
    filtered = [x for x in data if abs(x - mean) <= threshold * std_dev]
    return filtered


def sort_dataset(data):
    """
    Sort data in ascending order.
    
    Args:
        data: List of numeric values
    
    Returns:
        Sorted list of values
    """
    return sorted(data)


def merge_datasets(data1, data2):
    """
    Combine two datasets into a single list.
    
    Args:
        data1: First list of values
        data2: Second list of values
    
    Returns:
        Combined list containing all values from both datasets
    """
    return list(data1) + list(data2)


def validate_data(data):
    """
    Check if data is valid (not empty and contains only numbers).
    
    Args:
        data: List to validate
    
    Returns:
        True if data is valid, False otherwise
    """
    if not data:
        return False
    
    for item in data:
        if not isinstance(item, (int, float)):
            return False
    
    return True