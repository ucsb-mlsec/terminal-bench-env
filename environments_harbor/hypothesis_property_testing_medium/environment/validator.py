#!/usr/bin/env python3

import sys
sys.path.insert(0, '/workspace')

from validator import validate_email

def find_bugs():
    """Find concrete examples of validator bugs."""
    
    valid_rejected = []
    invalid_accepted = []
    
    # Test case 1: Valid emails with '+' character (commonly used for email aliases)
    # The validator rejects these because '+' is not in valid_chars
    test_email = "user+tag@example.com"
    if not validate_email(test_email):
        valid_rejected.append(test_email)
    
    # Test case 2: Valid emails with subdomain
    test_email = "test@mail.example.com"
    if validate_email(test_email):
        # This should be valid and is accepted, so not a bug
        pass
    else:
        valid_rejected.append(test_email)
    
    # Test case 3: Valid email with numbers in domain
    test_email = "admin@test123.com"
    if validate_email(test_email):
        # Should be valid and is accepted
        pass
    else:
        valid_rejected.append(test_email)
    
    # Test case 4: Valid email with hyphens
    test_email = "user-name@ex-ample.com"
    if validate_email(test_email):
        # Should be valid and is accepted
        pass
    else:
        valid_rejected.append(test_email)
    
    # Test case 5: More valid chars that might be rejected
    test_email = "user.name+tag@example.com"
    if not validate_email(test_email):
        valid_rejected.append(test_email)
    
    # Test case 6: Valid with underscore and plus
    test_email = "test_user+filter@domain.org"
    if not validate_email(test_email):
        valid_rejected.append(test_email)
    
    # Bug 2: Consecutive dots should be invalid but are accepted
    test_email = "user..name@example.com"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Bug 2 variation: Multiple consecutive dots
    test_email = "test...email@domain.com"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Bug 3: Leading dot in local part should be invalid
    test_email = ".user@example.com"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Bug 3: Trailing dot in local part should be invalid
    test_email = "user.@example.com"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Bug 4: Domain without dot should be invalid (no TLD)
    test_email = "user@localhost"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Bug 4 variation: Another domain without dot
    test_email = "admin@server"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Consecutive dots in domain should also be invalid
    test_email = "user@example..com"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Leading dot in domain
    test_email = "user@.example.com"
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    # Trailing dot in domain (before @)
    test_email = "user@example.com."
    if validate_email(test_email):
        invalid_accepted.append(test_email)
    
    return valid_rejected, invalid_accepted

def main():
    valid_rejected, invalid_accepted = find_bugs()
    
    # Write results to file
    with open('/workspace/bugs.txt', 'w') as f:
        # Write at least 2 of each type
        for email in valid_rejected[:2]:
            f.write(f"VALID_REJECTED: {email}\n")
        
        for email in invalid_accepted[:2]:
            f.write(f"INVALID_ACCEPTED: {email}\n")
        
        # Write any additional findings
        for email in valid_rejected[2:]:
            f.write(f"VALID_REJECTED: {email}\n")
        
        for email in invalid_accepted[2:]:
            f.write(f"INVALID_ACCEPTED: {email}\n")
    
    print(f"Found {len(valid_rejected)} valid emails incorrectly rejected")
    print(f"Found {len(invalid_accepted)} invalid emails incorrectly accepted")
    print("Results saved to /workspace/bugs.txt")

if __name__ == "__main__":
    main()