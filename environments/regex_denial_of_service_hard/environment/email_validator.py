#!/usr/bin/env python3

import re

def validate_email(email_string):
    """
    Validates email addresses using regex patterns.
    WARNING: Contains ReDoS vulnerabilities!
    """
    
    if not email_string or not isinstance(email_string, str):
        return False
    
    # Vulnerable pattern 1: Nested quantifiers in overall structure
    # Pattern attempts to match the entire email with overlapping groups
    overall_pattern = r'^([a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})+$'
    
    if not re.match(overall_pattern, email_string):
        return False
    
    # Split email into local and domain parts
    if '@' not in email_string:
        return False
    
    parts = email_string.split('@')
    if len(parts) != 2:
        return False
    
    local_part, domain_part = parts
    
    # Vulnerable pattern 2: Local part validation with catastrophic backtracking
    # Multiple overlapping quantifiers that can cause exponential behavior
    local_pattern = r'^([a-zA-Z0-9]+([._-]?[a-zA-Z0-9]+)*)+$'
    
    if not re.match(local_pattern, local_part):
        return False
    
    # Check for consecutive dots (invalid)
    if '..' in local_part:
        return False
    
    # Vulnerable pattern 3: Domain validation with nested quantifiers
    # Pattern has overlapping alternatives and nested quantifiers
    domain_pattern = r'^([a-zA-Z0-9]+([.-]?[a-zA-Z0-9]+)*)+\.[a-zA-Z]{2,}$'
    
    if not re.match(domain_pattern, domain_part):
        return False
    
    # Additional check: domain must have at least one dot
    if '.' not in domain_part:
        return False
    
    # Check that TLD (top-level domain) is at least 2 characters
    tld = domain_part.split('.')[-1]
    if len(tld) < 2:
        return False
    
    return True

if __name__ == '__main__':
    # Test cases
    valid_emails = [
        'user@example.com',
        'name.lastname@company.co.uk',
        'first_last@test-domain.org',
        'user123@example.com'
    ]
    
    invalid_emails = [
        'not-an-email',
        '@example.com',
        'user@',
        'user@domain',
        'user..name@example.com'
    ]
    
    print("Testing valid emails:")
    for email in valid_emails:
        print(f"  {email}: {validate_email(email)}")
    
    print("\nTesting invalid emails:")
    for email in invalid_emails:
        print(f"  {email}: {validate_email(email)}")