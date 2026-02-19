#!/usr/bin/env python3
"""
Utility functions and classes for the datautils package.
"""

import json
import os
from typing import Any, Dict


def format_output(data: Any, format_type: str) -> str:
    """
    Format data according to the specified format type.
    
    Args:
        data: The data to format
        format_type: The format type ('json', 'str', or 'repr')
    
    Returns:
        Formatted data as a string
    """
    if format_type == 'json':
        return json.dumps(data, indent=2)
    elif format_type == 'str':
        return str(data)
    elif format_type == 'repr':
        return repr(data)
    else:
        return str(data)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
    
    Returns:
        Dictionary containing configuration data
    """
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(data: Any, filepath: str) -> bool:
    """
    Save results to a JSON file.
    
    Args:
        data: The data to save
        filepath: Path to the output file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        return False


class Logger:
    """
    Simple logger class for logging messages at different levels.
    """
    
    def __init__(self, name: str = "datautils"):
        """
        Initialize the logger.
        
        Args:
            name: Name of the logger
        """
        self.name = name
    
    def info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message: The message to log
        """
        print(f"[INFO] {self.name}: {message}")
    
    def error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: The error message to log
        """
        print(f"[ERROR] {self.name}: {message}")
    
    def debug(self, message: str) -> None:
        """
        Log a debug message.
        
        Args:
            message: The debug message to log
        """
        print(f"[DEBUG] {self.name}: {message}")