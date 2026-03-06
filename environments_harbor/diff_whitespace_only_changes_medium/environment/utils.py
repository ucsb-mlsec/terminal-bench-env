#!/usr/bin/env python3
"""
Utility functions for the Python project.
"""


def calculate_sum(numbers):
    """
    Calculate the sum of a list of numbers.
    
    Args:
        numbers: A list of numeric values
        
    Returns:
        The sum of all numbers in the list
    """
    if not numbers:
        return 0
    return sum(numbers)


def format_string(text):
    """
    Format a string by converting it to uppercase.
    
    Args:
        text: The input string to format
        
    Returns:
        The uppercase version of the input string
    """
    if text is None:
        return ""
    return text.upper()


def validate_input(value):
    """
    Validate that the input value is not None.
    
    Args:
        value: The value to validate
        
    Returns:
        True if value is not None, False otherwise
    """
    return value is not None