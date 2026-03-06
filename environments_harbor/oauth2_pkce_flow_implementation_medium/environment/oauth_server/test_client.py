#!/usr/bin/env python3

import requests
import secrets
import hashlib
import base64
import urllib.parse
import sys

# Configuration
SERVER_URL = 'http://localhost:5000'
CLIENT_ID = 'test_client'
REDIRECT_URI = 'http://localhost:8080/callback'

def generate_pkce_pair():
    """
    Generate PKCE code_verifier and code_challenge pair.
    
    Returns:
        tuple: (code_verifier, code_challenge)
    """
    # Generate code_verifier: 43-128 characters
    code_verifier = secrets.token_urlsafe(64)  # Generates ~86 characters
    
    # Create code_challenge using S256 method
    # SHA256 hash the verifier
    challenge_bytes = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    
    # Base64url encode (no padding)
    code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')
    
    return code_verifier, code_challenge

def main():
    """
    Execute complete OAuth2 PKCE flow test.
    """
    try:
        print("=" * 60)
        print("Starting OAuth2 PKCE Flow Test")
        print("=" * 60)
        
        # Step 1: Generate PKCE pair
        print("\n[Step 1] Generating PKCE pair...")
        code_verifier, code_challenge = generate_pkce_pair()
        print(f"Code Verifier: {code_verifier}")
        print(f"Code Challenge: {code_challenge}")
        
        # Step 2: Build authorization URL
        print("\n[Step 2] Building authorization URL...")
        state = secrets.token_urlsafe(16)
        
        auth_params = {
            'client_id': CLIENT_ID,
            'redirect_uri': REDIRECT_URI,
            'response_type': 'code',
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        auth_url = f"{SERVER_URL}/authorize?{urllib.parse.urlencode(auth_params)}"
        print(f"Authorization URL: {auth_url}")
        
        # Step 3: Request authorization (simulate user authorization)
        print("\n[Step 3] Requesting authorization code...")
        auth_response = requests.get(auth_url, allow_redirects=False)
        
        if auth_response.status_code != 302:
            print(f"ERROR: Expected redirect (302), got {auth_response.status_code}")
            sys.exit(1)
        
        # Parse authorization code from redirect location
        redirect_location = auth_response.headers.get('Location', '')
        parsed_redirect = urllib.parse.urlparse(redirect_location)
        redirect_params = urllib.parse.parse_qs(parsed_redirect.query)
        
        if 'code' not in redirect_params:
            print(f"ERROR: No authorization code in redirect: {redirect_location}")
            sys.exit(1)
        
        authorization_code = redirect_params['code'][0]
        print(f"Authorization Code: {authorization_code}")
        
        # Step 4: Exchange authorization code for access token
        print("\n[Step 4] Exchanging authorization code for access token...")
        token_data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'code_verifier': code_verifier
        }
        
        token_response = requests.post(f"{SERVER_URL}/token", data=token_data)
        
        if token_response.status_code != 200:
            print(f"ERROR: Token request failed with status {token_response.status_code}")
            print(f"Response: {token_response.text}")
            sys.exit(1)
        
        token_json = token_response.json()
        print(f"Token Response: {token_json}")
        
        if 'access_token' not in token_json:
            print("ERROR: No access_token in response")
            sys.exit(1)
        
        access_token = token_json['access_token']
        print(f"Access Token: {access_token}")
        
        # Step 5: Access protected resource
        print("\n[Step 5] Accessing protected resource...")
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        resource_response = requests.get(f"{SERVER_URL}/resource", headers=headers)
        
        if resource_response.status_code != 200:
            print(f"ERROR: Resource request failed with status {resource_response.status_code}")
            print(f"Response: {resource_response.text}")
            sys.exit(1)
        
        resource_json = resource_response.json()
        print(f"Protected Resource Response: {resource_json}")
        
        # Success
        print("\n" + "=" * 60)
        print("PKCE Flow Test: SUCCESS")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Exception occurred during PKCE flow test")
        print(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)