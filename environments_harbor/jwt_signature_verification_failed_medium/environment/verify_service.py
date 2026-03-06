#!/usr/bin/env python3

import jwt
import sys
import os
from datetime import datetime

def verify_token(token, secret):
    """
    Verify a JWT token using the provided secret with HS256 algorithm.
    Returns True if valid, False otherwise.
    """
    try:
        # Verify token with HS256 algorithm
        decoded = jwt.decode(token, secret, algorithms=['HS256'])
        return True
    except jwt.ExpiredSignatureError:
        # Token has expired
        return False
    except jwt.InvalidTokenError:
        # Token signature is invalid or token is malformed
        return False
    except Exception as e:
        # Any other error during verification
        return False

def main():
    # Read the current secret (NEW secret only)
    # BUG: This only uses the new secret, not checking the old secret
    # This causes tokens signed with the old secret to fail verification
    try:
        with open('/app/secrets/current_secret.txt', 'r') as f:
            current_secret = f.read().strip()
    except Exception as e:
        print(f"Error reading secret: {e}")
        sys.exit(1)
    
    # Get token from command line or stdin
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = sys.stdin.read().strip()
    
    # Verify the token using ONLY the current secret
    # This is the problematic implementation - it should try both secrets
    if verify_token(token, current_secret):
        print("Token verification: SUCCESS")
        return 0
    else:
        print("Token verification: FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())