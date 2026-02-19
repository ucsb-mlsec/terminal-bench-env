#!/usr/bin/env python3

import requests
import time
import json
import sys

TOKEN_FILE = '/home/developer/service/.token'
LOG_FILE = '/home/developer/service/app.log'
API_URL = 'http://localhost:8200/v1/secret/db-creds'

def log_message(message):
    """Write a log message to the log file"""
    with open(LOG_FILE, 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def read_token():
    """Read the authentication token from file"""
    try:
        with open(TOKEN_FILE, 'r') as f:
            return f.read().strip()
    except Exception as e:
        log_message(f"ERROR: Failed to read token file: {e}")
        sys.exit(1)

def main():
    log_message("Service starting...")
    
    # Read initial token
    token = read_token()
    log_message("Token loaded successfully")
    
    # Main service loop
    while True:
        try:
            headers = {'X-Vault-Token': token}
            response = requests.get(API_URL, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                log_message(f"Successfully fetched credentials: {json.dumps(data)}")
            else:
                log_message(f"ERROR: API returned status code {response.status_code}")
                if response.status_code == 403:
                    log_message("ERROR: Token expired - service will fail")
                    break
            
            time.sleep(3)
            
        except Exception as e:
            log_message(f"ERROR: Request failed: {e}")
            time.sleep(3)

if __name__ == '__main__':
    main()