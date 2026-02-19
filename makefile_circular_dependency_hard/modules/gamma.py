"""Gamma module for mathematical and string operations."""

import math


def calculate_square_root(number):
    """Calculate the square root of a number."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)


def reverse_string(text):
    """Reverse a string."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return text[::-1]


def find_maximum(numbers):
    """Find the maximum value in a list of numbers."""
    if not numbers:
        return None
    return max(numbers)


def calculate_power(base, exponent):
    """Calculate base raised to the power of exponent."""
    return math.pow(base, exponent)