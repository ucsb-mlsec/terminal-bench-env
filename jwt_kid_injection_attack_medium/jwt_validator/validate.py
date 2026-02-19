#!/usr/bin/env python3

import jwt
import sys
import os
import json
import base64

def validate_token(token):
    try:
        # Decode header without verification to extract kid
        header = jwt.get_unverified_header(token)
        
        if 'kid' not in header:
            print("Error: No kid parameter in JWT header")
            return False
        
        kid = header['kid']
        
        # VULNERABLE: Direct concatenation allows path traversal
        keys_dir = '/home/agent/jwt_validator/keys/'
        key_path = keys_dir + kid
        
        # Load the key file
        if not os.path.exists(key_path):
            print(f"Error: Key file not found at {key_path}")
            return False
        
        with open(key_path, 'rb') as f:
            key = f.read()
        
        # Verify and decode the token
        payload = jwt.decode(token, key, algorithms=['HS256'])
        
        print("Token validated successfully!")
        print(f"Username: {payload.get('username')}")
        print(f"Role: {payload.get('role')}")
        return True
        
    except jwt.InvalidSignatureError:
        print("Error: Invalid signature")
        return False
    except jwt.ExpiredSignatureError:
        print("Error: Token has expired")
        return False
    except jwt.InvalidTokenError as e:
        print(f"Error: Invalid token - {str(e)}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 validate.py <jwt_token>")
        sys.exit(1)
    
    token = sys.argv[1]
    success = validate_token(token)
    sys.exit(0 if success else 1)