#!/usr/bin/env python3

import jwt
import os
import glob

def load_public_keys(directory_path):
    """
    Load all RSA public keys from PEM files in the specified directory.
    
    Args:
        directory_path: Path to directory containing .pem files
        
    Returns:
        List of public key strings
    """
    public_keys = []
    
    if not os.path.exists(directory_path):
        return public_keys
    
    pem_files = glob.glob(os.path.join(directory_path, "*.pem"))
    
    for pem_file in pem_files:
        try:
            with open(pem_file, 'r') as f:
                key_data = f.read()
                public_keys.append(key_data)
        except Exception as e:
            print(f"Error loading key from {pem_file}: {e}")
            continue
    
    return public_keys

def validate_token(token, keys_directory):
    """
    Validate a JWT token using RS256 algorithm against public keys in directory.
    
    Args:
        token: JWT token string to validate
        keys_directory: Path to directory containing public key PEM files
        
    Returns:
        True if token is valid, False otherwise
    """
    public_keys = load_public_keys(keys_directory)
    
    if not public_keys:
        print("No public keys found")
        return False
    
    # Try to validate token with each available public key
    for public_key in public_keys:
        try:
            # BUG #1: Using HS256 instead of RS256
            decoded = jwt.decode(token, public_key, algorithms=['HS256'])
            # If we get here, validation succeeded
            return True
        except jwt.InvalidSignatureError:
            # BUG #2: Breaking instead of continuing to try next key
            print("Signature validation failed")
            return False
        except jwt.ExpiredSignatureError:
            print("Token has expired")
            return False
        except jwt.InvalidTokenError as e:
            print(f"Invalid token: {e}")
            return False
        except Exception as e:
            print(f"Error validating token: {e}")
            return False
    
    return False