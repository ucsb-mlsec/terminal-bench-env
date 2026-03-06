#!/usr/bin/env python3
"""
Student sorting implementation for curriculum learning system.
Implements insertion sort with comparison tracking.
"""

class SortMetrics:
    """Tracks sorting performance metrics."""
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
    
    def reset(self):
        self.comparisons = 0
        self.swaps = 0

# Global metrics instance
metrics = SortMetrics()

def sort_array(arr):
    """Sorts an array of integers using insertion sort.
    
    This implementation represents a 'student' algorithm that works
    correctly but may not be optimally efficient for all cases.
    Tracks comparisons and swaps for performance analysis.
    
    Args:
        arr: List of integers to sort
        
    Returns:
        Sorted list of integers
        
    Time Complexity: O(n^2) worst case, O(n) best case
    Space Complexity: O(1)
    """
    # Reset metrics for this sort
    metrics.reset()
    
    # Handle edge cases
    if not arr or len(arr) <= 1:
        return arr.copy() if arr else []
    
    # Create a copy to avoid modifying original
    result = arr.copy()
    n = len(result)
    
    # Insertion sort implementation
    for i in range(1, n):
        key = result[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0:
            metrics.comparisons += 1
            if result[j] > key:
                result[j + 1] = result[j]
                metrics.swaps += 1
                j -= 1
            else:
                break
        
        result[j + 1] = key
    
    return result

def get_metrics():
    """Returns the current sorting metrics.
    
    Returns:
        Dictionary with comparisons and swaps counts
    """
    return {
        'comparisons': metrics.comparisons,
        'swaps': metrics.swaps
    }