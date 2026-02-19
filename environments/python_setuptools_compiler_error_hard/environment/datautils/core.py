"""
Core data utility functions for the datautils package.
"""


class DataProcessor:
    """A class for processing data."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.initialized = True
    
    def process(self, data):
        """
        Process the given data.
        
        Args:
            data: The data to process
            
        Returns:
            The processed data
        """
        return data


def clean_data(data):
    """
    Clean the given data.
    
    Args:
        data: The data to clean
        
    Returns:
        The cleaned data
    """
    if data is None:
        return []
    return data


def merge_datasets(data1, data2):
    """
    Merge two datasets together.
    
    Args:
        data1: The first dataset
        data2: The second dataset
        
    Returns:
        The merged dataset
    """
    if isinstance(data1, list) and isinstance(data2, list):
        return data1 + data2
    elif isinstance(data1, dict) and isinstance(data2, dict):
        result = data1.copy()
        result.update(data2)
        return result
    return [data1, data2]


def calculate_statistics(numbers):
    """
    Calculate statistics for a list of numbers.
    
    Args:
        numbers: A list of numbers
        
    Returns:
        A dictionary with 'mean', 'max', 'min' keys
    """
    if not numbers:
        return {'mean': 0, 'max': 0, 'min': 0}
    
    return {
        'mean': sum(numbers) / len(numbers),
        'max': max(numbers),
        'min': min(numbers)
    }