#!/usr/bin/env python3

import time


class APIClient:
    """Simulates an OAuth2-authenticated API client."""
    
    def __init__(self, token_manager):
        """
        Initialize the API client with a token manager.
        
        Args:
            token_manager: An instance of TokenManager
        """
        self.token_manager = token_manager
    
    def fetch_data(self):
        """
        Simulate fetching data from an OAuth2-protected API.
        
        Returns:
            dict: A dictionary with success status and data
            
        Raises:
            Exception: If authentication fails (401 Unauthorized)
        """
        # Get the current access token
        token_data = self.token_manager.get_token()
        
        if not token_data:
            raise Exception('401 Unauthorized: No token available')
        
        access_token = token_data.get('access_token')
        expires_at = token_data.get('expires_at')
        
        if not access_token or not expires_at:
            raise Exception('401 Unauthorized: Invalid token data')
        
        # Check if token is expired
        current_time = time.time()
        if current_time >= expires_at:
            raise Exception('401 Unauthorized: Token expired')
        
        # Token is valid, return successful response
        return {
            'success': True,
            'data': 'sample data'
        }