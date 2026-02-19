#!/usr/bin/env python3
"""
Genomic Statistics Module

This module provides statistical analysis functions for genomic data,
including coverage statistics, quality score analysis, and variance calculations.
"""

def mean_coverage(coverage_data):
    """
    Calculate the mean coverage from a list of coverage values.
    
    Args:
        coverage_data: List or iterable of coverage values
        
    Returns:
        float: Mean coverage value
    """
    if not coverage_data:
        return 0.0
    return sum(coverage_data) / len(coverage_data)


def coverage_distribution(coverage_data, bins=10):
    """
    Calculate the distribution of coverage values across bins.
    
    Args:
        coverage_data: List of coverage values
        bins: Number of bins for the histogram
        
    Returns:
        dict: Distribution of coverage values
    """
    if not coverage_data:
        return {}
    
    min_val = min(coverage_data)
    max_val = max(coverage_data)
    bin_width = (max_val - min_val) / bins if max_val > min_val else 1
    
    distribution = {i: 0 for i in range(bins)}
    for value in coverage_data:
        bin_index = min(int((value - min_val) / bin_width), bins - 1)
        distribution[bin_index] += 1
    
    return distribution


def average_quality(quality_scores):
    """
    Calculate the average quality score from a list of quality values.
    
    Args:
        quality_scores: List of quality score values
        
    Returns:
        float: Average quality score
    """
    if not quality_scores:
        return 0.0
    return sum(quality_scores) / len(quality_scores)


def quality_histogram(quality_scores, threshold=30):
    """
    Generate a histogram of quality scores above and below a threshold.
    
    Args:
        quality_scores: List of quality scores
        threshold: Quality threshold value
        
    Returns:
        dict: Count of scores above and below threshold
    """
    above = sum(1 for q in quality_scores if q >= threshold)
    below = sum(1 for q in quality_scores if q < threshold)
    return {'above_threshold': above, 'below_threshold': below}


def calculate_variance(data):
    """
    Calculate the variance of a dataset.
    
    Args:
        data: List of numerical values
        
    Returns:
        float: Variance of the data
    """
    if len(data) < 2:
        return 0.0
    
    mean = sum(data) / len(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diffs) / len(data)
    return variance


def statistical_summary(data):
    """
    Generate a comprehensive statistical summary of the data.
    
    Args:
        data: List of numerical values
        
    Returns:
        dict: Dictionary containing mean, variance, min, max, and count
    """
    if not data:
        return {'mean': 0, 'variance': 0, 'min': 0, 'max': 0, 'count': 0}
    
    return {
        'mean': sum(data) / len(data),
        'variance': calculate_variance(data),
        'min': min(data),
        'max': max(data),
        'count': len(data)
    }