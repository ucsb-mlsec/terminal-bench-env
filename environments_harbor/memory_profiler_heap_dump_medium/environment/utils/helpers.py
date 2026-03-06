#!/usr/bin/env python3
"""
Utility helper functions for the web application.
Provides common functionality for formatting, validation, and response generation.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def format_timestamp(timestamp: Union[int, float, datetime]) -> str:
    """
    Format a timestamp into a standardized ISO 8601 string.
    
    Args:
        timestamp: Unix timestamp (int/float) or datetime object
        
    Returns:
        Formatted timestamp string in ISO 8601 format
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = timestamp
    return dt.isoformat()


def validate_request_data(data: Dict[str, Any]) -> bool:
    """
    Validate incoming request data for required fields.
    
    Args:
        data: Dictionary containing request data
        
    Returns:
        True if validation passes, False otherwise
    """
    if not isinstance(data, dict):
        return False
    required_fields = ['action', 'user_id']
    return all(field in data for field in required_fields)


def generate_response(status: str, message: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a standardized API response dictionary.
    
    Args:
        status: Response status ('success' or 'error')
        message: Human-readable message
        data: Optional data payload
        
    Returns:
        Dictionary containing standardized response structure
    """
    response = {
        'status': status,
        'message': message,
        'timestamp': format_timestamp(datetime.now())
    }
    if data is not None:
        response['data'] = data
    return response


def calculate_metrics(data_points: List[Union[int, float]]) -> Dict[str, float]:
    """
    Calculate basic statistical metrics from a list of data points.
    
    Args:
        data_points: List of numerical values
        
    Returns:
        Dictionary containing calculated metrics (average, min, max, total)
    """
    if not data_points:
        return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'total': 0.0}
    
    total = sum(data_points)
    return {
        'average': total / len(data_points),
        'min': min(data_points),
        'max': max(data_points),
        'total': total
    }