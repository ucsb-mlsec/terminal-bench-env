#!/usr/bin/env python3

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Parsed JSON data as dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")
        raise


def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Write data to a JSON file.
    
    Args:
        data: Dictionary to write
        file_path: Output file path
        indent: JSON indentation level
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (IOError, TypeError) as e:
        logging.error(f"Error writing JSON file {file_path}: {e}")
        return False


def format_timestamp(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object as a string.
    
    Args:
        dt: Datetime object (defaults to now)
        fmt: Format string
    
    Returns:
        Formatted datetime string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)


def validate_path(path: Union[str, Path], must_exist: bool = True) -> bool:
    """
    Validate if a path exists and is accessible.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
    
    Returns:
        True if valid, False otherwise
    """
    p = Path(path)
    if must_exist:
        return p.exists()
    return p.parent.exists()


def sanitize_string(text: str, max_length: int = 255, allowed_chars: str = "") -> str:
    """
    Sanitize a string by removing invalid characters.
    
    Args:
        text: Input string
        max_length: Maximum allowed length
        allowed_chars: Additional allowed characters
    
    Returns:
        Sanitized string
    """
    import re
    pattern = r'[^a-zA-Z0-9_\-\.' + re.escape(allowed_chars) + r']'
    sanitized = re.sub(pattern, '_', text)
    return sanitized[:max_length]


def ensure_directory(dir_path: Union[str, Path], mode: int = 0o755) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Directory path
        mode: Directory permissions
    
    Returns:
        True if directory exists or was created
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True, mode=mode)
        return True
    except OSError as e:
        logging.error(f"Error creating directory {dir_path}: {e}")
        return False