import requests
import json
from typing import Dict, Any, Optional


class APIClient:
    """Client for making API requests."""
    
    def __init__(self, base_url: str):
        """Initialize the API client with a base URL.
        
        Args:
            base_url: The base URL for the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get(self, endpoint: str) -> requests.Response:
        """Make a GET request to the specified endpoint.
        
        Args:
            endpoint: The API endpoint to request
            
        Returns:
            Response object from the request
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.get(url)
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> requests.Response:
        """Make a POST request to the specified endpoint.
        
        Args:
            endpoint: The API endpoint to post to
            data: Dictionary of data to send in the request body
            
        Returns:
            Response object from the request
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.post(url, json=data)
    
    def delete(self, endpoint: str) -> requests.Response:
        """Make a DELETE request to the specified endpoint.
        
        Args:
            endpoint: The API endpoint to delete
            
        Returns:
            Response object from the request
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.delete(url)


def create_client(url: str) -> APIClient:
    """Factory function to create an APIClient instance.
    
    Args:
        url: The base URL for the API client
        
    Returns:
        A new APIClient instance
    """
    return APIClient(url)


def handle_response(response: requests.Response) -> Optional[Dict[str, Any]]:
    """Process and handle API responses.
    
    Args:
        response: The response object from an API request
        
    Returns:
        Parsed JSON data if successful, None otherwise
    """
    if response.status_code == 200:
        return response.json()
    return None