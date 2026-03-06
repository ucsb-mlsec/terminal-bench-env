#!/usr/bin/env python3

import requests
import hashlib
import base64
import secrets
import json
import sys


def base64url_encode(data):
    """Encode bytes to base64url format (no padding)"""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def generate_code_verifier():
    """Generate a random code verifier (43-128 characters)"""
    return base64url_encode(secrets.token_bytes(32))


def create_code_challenge(verifier):
    """Create S256 code challenge from verifier"""
    digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64url_encode(digest)


def main():
    try:
        # Step 1: Generate code verifier and challenge
        code_verifier = generate_code_verifier()
        code_challenge = create_code_challenge(code_verifier)
        
        print(f"Generated code_verifier: {code_verifier}")
        print(f"Generated code_challenge: {code_challenge}")
        
        # Step 2: Request authorization code
        print("\nRequesting authorization code...")
        auth_response = requests.get(
            'http://localhost:5000/authorize',
            params={
                'client_id': 'mobile_app',
                'response_type': 'code',
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256'
            }
        )
        
        if auth_response.status_code != 200:
            print(f"Authorization request failed: {auth_response.status_code}")
            print(f"Response: {auth_response.text}")
            sys.exit(1)
        
        auth_data = auth_response.json()
        if 'authorization_code' not in auth_data:
            print("No authorization_code in response")
            print(f"Response: {auth_data}")
            sys.exit(1)
        
        authorization_code = auth_data['authorization_code']
        print(f"Received authorization_code: {authorization_code}")
        
        # Step 3: Exchange authorization code for access token
        print("\nExchanging code for access token...")
        token_response = requests.post(
            'http://localhost:5000/token',
            data={
                'grant_type': 'authorization_code',
                'code': authorization_code,
                'code_verifier': code_verifier
            }
        )
        
        if token_response.status_code != 200:
            print(f"Token request failed: {token_response.status_code}")
            print(f"Response: {token_response.text}")
            sys.exit(1)
        
        token_data = token_response.json()
        if 'access_token' not in token_data:
            print("No access_token in response")
            print(f"Response: {token_data}")
            sys.exit(1)
        
        access_token = token_data['access_token']
        print(f"Successfully received access_token: {access_token}")
        print("\n✓ PKCE flow completed successfully!")
        
        sys.exit(0)
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()