"""Delta module for text and numeric operations."""


def count_vowels(text):
    """Count the number of vowels in a given text string.
    
    Args:
        text: String to analyze
        
    Returns:
        Integer count of vowels
    """
    vowels = 'aeiouAEIOU'
    return sum(1 for char in text if char in vowels)


def average(numbers):
    """Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numeric values
        
    Returns:
        Float average, or 0 if list is empty
    """
    return sum(numbers) / len(numbers) if numbers else 0


def filter_evens(numbers):
    """Filter even numbers from a list.
    
    Args:
        numbers: List of integers
        
    Returns:
        List containing only even numbers
    """
    return [n for n in numbers if n % 2 == 0]