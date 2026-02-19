#!/usr/bin/env python3
"""
Request Handler Module

This module provides functionality for handling HTTP requests in the web application.
It includes validation, parsing, and processing of incoming requests.
"""

import json
import logging
from flask import Request, Response
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class RequestHandler:
    """
    Handles HTTP request processing, validation, and parsing.
    
    This class provides methods to process incoming HTTP requests,
    validate their structure and content, and extract relevant data.
    """
    
    def __init__(self):
        """Initialize the RequestHandler with default configuration."""
        self.max_content_length = 10 * 1024 * 1024  # 10MB
        self.allowed_content_types = ['application/json', 'application/x-www-form-urlencoded']
        logger.info("RequestHandler initialized")
    
    def process_request(self, request: Request) -> Dict[str, Any]:
        """
        Process an incoming HTTP request.
        
        Args:
            request: Flask Request object
            
        Returns:
            Dictionary containing processed request data
        """
        # Allocate dictionary for request metadata
        request_data = {
            'method': request.method,
            'path': request.path,
            'timestamp': None
        }
        
        # Validate the request
        if not self.validate_input(request):
            logger.warning(f"Invalid request received: {request.path}")
            return {'error': 'Invalid request format'}
        
        # Parse headers
        headers = self.parse_headers(request)
        request_data['headers'] = headers
        
        # Parse body if present
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = self.parse_body(request)
                request_data['body'] = body
            except Exception as e:
                logger.error(f"Error parsing request body: {e}")
                request_data['body'] = None
        
        logger.debug(f"Successfully processed request: {request.path}")
        return request_data
    
    def validate_input(self, request: Request) -> bool:
        """
        Validate incoming request structure and content.
        
        Args:
            request: Flask Request object
            
        Returns:
            True if request is valid, False otherwise
        """
        # Check content length
        if request.content_length and request.content_length > self.max_content_length:
            logger.warning(f"Request exceeds maximum content length: {request.content_length}")
            return False
        
        # Check content type for POST/PUT requests
        if request.method in ['POST', 'PUT'] and request.content_type:
            if not any(ct in request.content_type for ct in self.allowed_content_types):
                logger.warning(f"Unsupported content type: {request.content_type}")
                return False
        
        return True
    
    def parse_headers(self, request: Request) -> Dict[str, str]:
        """
        Parse and extract relevant headers from the request.
        
        Args:
            request: Flask Request object
            
        Returns:
            Dictionary of parsed headers
        """
        headers = {}
        
        # Extract common headers
        if request.headers.get('User-Agent'):
            headers['user_agent'] = request.headers.get('User-Agent')
        
        if request.headers.get('Content-Type'):
            headers['content_type'] = request.headers.get('Content-Type')
        
        if request.headers.get('Authorization'):
            headers['authorization'] = 'Bearer ***'  # Mask sensitive data
        
        return headers
    
    def parse_body(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Parse request body based on content type.
        
        Args:
            request: Flask Request object
            
        Returns:
            Parsed body data as dictionary, or None if parsing fails
        """
        try:
            if 'application/json' in request.content_type:
                return request.get_json()
            elif 'application/x-www-form-urlencoded' in request.content_type:
                return dict(request.form)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            return None