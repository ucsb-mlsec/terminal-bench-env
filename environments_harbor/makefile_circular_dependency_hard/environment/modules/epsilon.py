"""Epsilon module for validation and list operations."""

def is_palindrome(text):
    """Check if a string is a palindrome, ignoring spaces and case."""
    clean = text.replace(' ', '').lower()
    return clean == clean[::-1]

def sum_list(numbers):
    """Calculate the sum of all numbers in a list."""
    if not numbers:
        return 0
    return sum(numbers)

def generate_range(start, end, step=1):
    """Generate a list of numbers from start to end with given step."""
    return list(range(start, end, step))

def average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def main():
    """Test the epsilon module functions."""
    print("Testing epsilon module...")
    print(f"is_palindrome('racecar'): {is_palindrome('racecar')}")
    print(f"sum_list([1, 2, 3, 4, 5]): {sum_list([1, 2, 3, 4, 5])}")
    print(f"generate_range(0, 10, 2): {generate_range(0, 10, 2)}")
    print(f"average([10, 20, 30]): {average([10, 20, 30])}")

if __name__ == '__main__':
    main()