#!/usr/bin/env python3

import sys
import json
import configparser
import time

def check_key(api_key):
    # Read configuration
    config = configparser.ConfigParser()
    config.read('/opt/auth/config.ini')
    
    grace_period_days = int(config.get('rotation', 'grace_period_days'))
    
    # Read keys from JSON file
    with open('/opt/auth/keys.json', 'r') as f:
        keys = json.load(f)
    
    # Find the key in the array
    key_found = None
    for key_obj in keys:
        if key_obj['key'] == api_key:
            key_found = key_obj
            break
    
    if not key_found:
        print('INVALID')
        return
    
    # Check if key is active
    if key_found.get('active', False):
        print('VALID')
        return
    
    # Check if key is within grace period
    current_time = int(time.time())
    key_age_seconds = current_time - key_found['created']
    key_age_days = key_age_seconds / 86400
    
    if key_age_days <= grace_period_days
        print('VALID')
    else:
        print('INVALID')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: check_key.py <api_key>')
        sys.exit(1)
    
    check_key(sys.argv[1])