#!/usr/bin/env python3
"""
Utility functions for the project.

This module contains various helper functions used throughout the application.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path


def read_config(config_path):
    """
    Read and parse a JSON configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Parsed configuration dictionary
        
    # TODO: Add type hints
    # TODO: Add logging
    """
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def write_config(config_path, config_data):
    """
    Write configuration data to a JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        config_data (dict): Configuration data to write
        
    # TODO: Handle edge cases
    # TODO: Add validation for config_data
    """
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)


def calculate_file_hash(file_path, algorithm='sha256'):
    """
    Calculate the hash of a file using the specified algorithm.
    
    Args:
        file_path (str): Path to the file
        algorithm (str): Hash algorithm to use (default: sha256)
        
    Returns:
        str: Hexadecimal hash string
        
    # TODO: Add support for more hash algorithms
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_timestamp():
    """
    Get the current timestamp in ISO format.
    
    Returns:
        str: ISO formatted timestamp
    """
    return datetime.now().isoformat()


def ensure_directory(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        bool: True if directory exists or was created successfully
        
    # TODO: Add error handling for permission issues
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory: {e}", file=sys.stderr)
        return False


def parse_command_line_args(args):
    """
    Parse command line arguments into a dictionary.
    
    Args:
        args (list): List of command line arguments
        
    Returns:
        dict: Parsed arguments as key-value pairs
        
    # TODO: Refactor this function to use argparse
    # TODO: Add validation for argument formats
    """
    parsed = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            parsed[key.lstrip('-')] = value
    return parsed


def format_bytes(bytes_value):
    """
    Format bytes into a human-readable string.
    
    Args:
        bytes_value (int): Number of bytes
        
    Returns:
        str: Formatted string with appropriate unit (B, KB, MB, GB, TB)
        
    # TODO: Add support for custom precision
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def list_files_recursive(directory, extension=None):
    """
    Recursively list all files in a directory.
    
    Args:
        directory (str): Root directory to search
        extension (str, optional): Filter by file extension
        
    Returns:
        list: List of file paths
        
    # TODO: Handle edge cases for symlinks
    # TODO: Add option to exclude hidden files
    """
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if extension is None or filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    # Example usage
    print("Utility functions loaded successfully")
    print(f"Current timestamp: {get_timestamp()}")