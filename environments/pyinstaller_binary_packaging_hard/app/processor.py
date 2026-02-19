"""Text transformation utilities.

This module provides simple text transformation functions including
uppercase conversion, lowercase conversion, and character reversal.
"""


def uppercase(text):
    """Convert text to uppercase.
    
    Args:
        text: A string to convert
        
    Returns:
        The input string converted to uppercase
    """
    return text.upper()


def lowercase(text):
    """Convert text to lowercase.
    
    Args:
        text: A string to convert
        
    Returns:
        The input string converted to lowercase
    """
    return text.lower()


def reverse(text):
    """Reverse the characters in text.
    
    Args:
        text: A string to reverse
        
    Returns:
        The input string with characters in reverse order
    """
    return text[::-1]