"""Beta module for string and math operations."""


def multiply(a, b):
    """Multiply two numbers and return the result."""
    return a * b


def to_upper(text):
    """Convert a string to uppercase."""
    return text.upper()


def factorial(n):
    """Calculate factorial of a number recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def power(base, exponent):
    """Calculate base raised to the power of exponent."""
    return base ** exponent


def reverse_string(text):
    """Reverse a string."""
    return text[::-1]


if __name__ == "__main__":
    print("Beta module loaded successfully")
    print(f"multiply(5, 3) = {multiply(5, 3)}")
    print(f"to_upper('hello') = {to_upper('hello')}")
    print(f"factorial(5) = {factorial(5)}")