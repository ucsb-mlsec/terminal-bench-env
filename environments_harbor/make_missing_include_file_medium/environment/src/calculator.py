#!/usr/bin/env python3
"""
Calculator module providing basic arithmetic operations.
"""


def add(a, b):
    """
    Add two numbers together.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
    
    Returns:
        The sum of a and b
    """
    return a + b


def subtract(a, b):
    """
    Subtract b from a.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
    
    Returns:
        The difference of a minus b
    """
    return a - b


def multiply(a, b):
    """
    Multiply two numbers together.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
    
    Returns:
        The product of a and b
    """
    return a * b


def divide(a, b):
    """
    Divide a by b.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
    
    Returns:
        The quotient of a divided by b
    
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b