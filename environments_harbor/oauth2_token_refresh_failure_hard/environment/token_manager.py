#!/usr/bin/env python3

import json
import time
import os
import random
import fcntl

class TokenManager:
    def __init__(self, token_file_path='tokens.json'):
        self.token_file_path = token_file_path
        
    def get_token(self):
        """Get current access token, refreshing if expired"""
        tokens = self.load_tokens()
        
        if not tokens:
            return None
            
        current_time = time.time()
        expires_at = tokens.get('expires_at', 0)
        
        if current_time >= expires_at:
            return self.refresh_token()
        
        return tokens.get('access_token')
    
    def refresh_token(self):
        """Refresh the access token"""
        # Simulate OAuth2 API call delay
        time.sleep(0.1)
        
        # Use file locking to prevent race conditions
        with open(self.token_file_path, 'r+') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            try:
                # Read current tokens
                f.seek(0)
                tokens = json.load(f)
                
                # Generate new access token
                new_access_token = f"token_{time.time()}_{random.randint(1000, 9999)}"
                
                # Update token data
                tokens['access_token'] = new_access_token
                tokens['expires_at'] = time.time() + 3600
                
                # Write updated tokens back
                f.seek(0)
                f.truncate()
                json.dump(tokens, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
                
                return new_access_token
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def save_tokens(self, tokens):
        """Write token data to JSON file with locking"""
        with open(self.token_file_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(tokens, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def load_tokens(self):
        """Read token data from JSON file with locking"""
        if not os.path.exists(self.token_file_path):
            return {}
        
        try:
            with open(self.token_file_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    tokens = json.load(f)
                    return tokens
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError):
            return {}