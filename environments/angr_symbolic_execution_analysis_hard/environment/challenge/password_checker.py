#!/usr/bin/env python3

import sys
import hashlib

def validate_password(password):
    """
    Validates password against multiple constraints.
    Returns True if all constraints are satisfied, False otherwise.
    """
    
    # Constraint 1: Length must be exactly 18 characters
    if len(password) != 18:
        return False
    
    # Constraint 2: Must contain at least 2 uppercase letters
    uppercase_count = sum(1 for c in password if c.isupper())
    if uppercase_count < 2:
        return False
    
    # Constraint 3: Must contain at least 3 lowercase letters
    lowercase_count = sum(1 for c in password if c.islower())
    if lowercase_count < 3:
        return False
    
    # Constraint 4: Must contain at least 3 digits
    digit_count = sum(1 for c in password if c.isdigit())
    if digit_count < 3:
        return False
    
    # Constraint 5: Must contain at least 2 special characters
    special_count = sum(1 for c in password if not c.isalnum())
    if special_count < 2:
        return False
    
    # Constraint 6: First character must be uppercase
    if not password[0].isupper():
        return False
    
    # Constraint 7: Last character must be a digit
    if not password[-1].isdigit():
        return False
    
    # Constraint 8: Character at position 5 must be an underscore
    if password[5] != '_':
        return False
    
    # Constraint 9: Character at position 10 must be '@'
    if password[10] != '@':
        return False
    
    # Constraint 10: Sum of ASCII values of characters at positions 0, 1, 2 must equal 234
    if ord(password[0]) + ord(password[1]) + ord(password[2]) != 234:
        return False
    
    # Constraint 11: ASCII value at position 7 must equal ASCII at position 8 minus 1
    if ord(password[7]) != ord(password[8]) - 1:
        return False
    
    # Constraint 12: Product of digits must be divisible by 6
    digits = [int(c) for c in password if c.isdigit()]
    product = 1
    for d in digits:
        if d == 0:
            return False  # Avoid zero in product
        product *= d
    if product % 6 != 0:
        return False
    
    # Constraint 13: Character at position 3 must be lowercase 'a'
    if password[3] != 'a':
        return False
    
    # Constraint 14: Sum of all ASCII values must be divisible by 17
    total_ascii = sum(ord(c) for c in password)
    if total_ascii % 17 != 0:
        return False
    
    # Constraint 15: Character at position 15 must be 's' or 'S'
    if password[15].lower() != 's':
        return False
    
    # Constraint 16: Characters at positions 6 and 14 must be digits
    if not (password[6].isdigit() and password[14].isdigit()):
        return False
    
    # Constraint 17: ASCII value at position 4 must be even
    if ord(password[4]) % 2 != 0:
        return False
    
    # Constraint 18: The digit at position 17 must be between 3 and 7
    if not password[17].isdigit() or int(password[17]) < 3 or int(password[17]) > 7:
        return False
    
    # Constraint 19: Sum of digits at even positions must equal sum of digits at odd positions (if they exist)
    even_pos_digits = [int(password[i]) for i in range(len(password)) if i % 2 == 0 and password[i].isdigit()]
    odd_pos_digits = [int(password[i]) for i in range(len(password)) if i % 2 == 1 and password[i].isdigit()]
    if sum(even_pos_digits) != sum(odd_pos_digits):
        return False
    
    # Constraint 20: Character at position 12 must be a lowercase letter
    if not password[12].islower():
        return False
    
    return True


def main():
    if len(sys.argv) != 2:
        print("ACCESS DENIED")
        sys.exit(1)
    
    password = sys.argv[1]
    
    if validate_password(password):
        print("ACCESS GRANTED")
        sys.exit(0)
    else:
        print("ACCESS DENIED")
        sys.exit(1)


if __name__ == "__main__":
    main()