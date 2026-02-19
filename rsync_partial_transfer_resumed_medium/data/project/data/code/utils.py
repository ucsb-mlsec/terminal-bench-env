#!/usr/bin/env python3
"""
Utility Module - Comprehensive collection of helper functions
Contains utilities for string manipulation, date handling, file operations,
data conversion, validation, and common operations.
"""

import os
import re
import sys
import json
import hashlib
import datetime
import time
import random
import string
import base64
import urllib.parse
import csv
import collections
from typing import (
    Any, Dict, List, Optional, Union, Tuple, Callable, 
    TypeVar, Generic, Iterable, Set
)
from pathlib import Path
from decimal import Decimal
from functools import wraps
import math


T = TypeVar('T')


# String Manipulation Utilities
# =============================

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, adding a suffix if truncated.
    
    Args:
        text: The string to truncate
        max_length: Maximum length including suffix
        suffix: String to append if truncated
        
    Returns:
        Truncated string with suffix if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def camel_to_snake(text: str) -> str:
    """
    Convert camelCase string to snake_case.
    
    Args:
        text: String in camelCase format
        
    Returns:
        String in snake_case format
    """
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    return pattern.sub('_', text).lower()


def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
    """
    Convert snake_case string to camelCase.
    
    Args:
        text: String in snake_case format
        capitalize_first: Whether to capitalize first letter
        
    Returns:
        String in camelCase format
    """
    components = text.split('_')
    if capitalize_first:
        return ''.join(x.title() for x in components)
    return components[0] + ''.join(x.title() for x in components[1:])


def pluralize(word: str, count: int) -> str:
    """
    Simple pluralization of English words.
    
    Args:
        word: The word to pluralize
        count: Number determining if plural needed
        
    Returns:
        Pluralized word if count != 1
    """
    if count == 1:
        return word
    
    irregular = {
        'person': 'people',
        'child': 'children',
        'foot': 'feet',
        'tooth': 'teeth',
        'man': 'men',
        'woman': 'women',
        'mouse': 'mice',
    }
    
    if word.lower() in irregular:
        return irregular[word.lower()]
    
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    elif word.endswith('y') and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    elif word.endswith('f'):
        return word[:-1] + 'ves'
    elif word.endswith('fe'):
        return word[:-2] + 'ves'
    else:
        return word + 's'


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in a string (collapse multiple spaces, trim).
    
    Args:
        text: String to normalize
        
    Returns:
        String with normalized whitespace
    """
    return ' '.join(text.split())


def remove_accents(text: str) -> str:
    """
    Remove accents from characters in a string.
    
    Args:
        text: String with accented characters
        
    Returns:
        String with accents removed
    """
    import unicodedata
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def is_palindrome(text: str, ignore_case: bool = True, ignore_spaces: bool = True) -> bool:
    """
    Check if a string is a palindrome.
    
    Args:
        text: String to check
        ignore_case: Whether to ignore case differences
        ignore_spaces: Whether to ignore spaces
        
    Returns:
        True if string is palindrome
    """
    processed = text
    if ignore_spaces:
        processed = processed.replace(' ', '')
    if ignore_case:
        processed = processed.lower()
    return processed == processed[::-1]


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from a string.
    
    Args:
        text: String containing numbers
        
    Returns:
        List of extracted numbers
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]


def mask_sensitive_data(text: str, visible_chars: int = 4, mask_char: str = '*') -> str:
    """
    Mask sensitive data showing only last few characters.
    
    Args:
        text: String to mask
        visible_chars: Number of characters to keep visible
        mask_char: Character to use for masking
        
    Returns:
        Masked string
    """
    if len(text) <= visible_chars:
        return mask_char * len(text)
    return mask_char * (len(text) - visible_chars) + text[-visible_chars:]


def generate_slug(text: str, max_length: int = 50) -> str:
    """
    Generate a URL-friendly slug from text.
    
    Args:
        text: Text to convert to slug
        max_length: Maximum length of slug
        
    Returns:
        URL-friendly slug
    """
    slug = text.lower()
    slug = remove_accents(slug)
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '-', slug)
    slug = slug.strip('-')
    return slug[:max_length]


# Date and Time Utilities
# ========================

def parse_date(date_string: str, formats: Optional[List[str]] = None) -> Optional[datetime.datetime]:
    """
    Parse a date string trying multiple formats.
    
    Args:
        date_string: Date string to parse
        formats: List of format strings to try
        
    Returns:
        Datetime object or None if parsing fails
    """
    if formats is None:
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
        ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    return None


def get_date_range(start_date: datetime.date, end_date: datetime.date) -> List[datetime.date]:
    """
    Generate list of dates between start and end dates.
    
    Args:
        start_date: Starting date
        end_date: Ending date
        
    Returns:
        List of dates in range
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += datetime.timedelta(days=1)
    return dates


def is_business_day(date: datetime.date) -> bool:
    """
    Check if a date is a business day (Monday-Friday).
    
    Args:
        date: Date to check
        
    Returns:
        True if business day
    """
    return date.weekday() < 5


def get_next_business_day(date: datetime.date) -> datetime.date:
    """
    Get the next business day after given date.
    
    Args:
        date: Starting date
        
    Returns:
        Next business day
    """
    next_day = date + datetime.timedelta(days=1)
    while not is_business_day(next_day):
        next_day += datetime.timedelta(days=1)
    return next_day


def format_duration(seconds: int) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h"


def get_age(birth_date: datetime.date, reference_date: Optional[datetime.date] = None) -> int:
    """
    Calculate age from birth date.
    
    Args:
        birth_date: Date of birth
        reference_date: Date to calculate age at (default: today)
        
    Returns:
        Age in years
    """
    if reference_date is None:
        reference_date = datetime.date.today()
    
    age = reference_date.year - birth_date.year
    if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def get_quarter(date: datetime.date) -> int:
    """
    Get fiscal quarter for a date.
    
    Args:
        date: Date to get quarter for
        
    Returns:
        Quarter number (1-4)
    """
    return (date.month - 1) // 3 + 1


def add_business_days(start_date: datetime.date, days: int) -> datetime.date:
    """
    Add business days to a date.
    
    Args:
        start_date: Starting date
        days: Number of business days to add
        
    Returns:
        Resulting date
    """
    current = start_date
    days_added = 0
    
    while days_added < days:
        current += datetime.timedelta(days=1)
        if is_business_day(current):
            days_added += 1
    
    return current


def get_week_number(date: datetime.date) -> int:
    """
    Get ISO week number for a date.
    
    Args:
        date: Date to get week number for
        
    Returns:
        Week number
    """
    return date.isocalendar()[1]


# File Operation Utilities
# =========================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(path: Union[str, Path], human_readable: bool = False) -> Union[int, str]:
    """
    Get file size in bytes or human-readable format.
    
    Args:
        path: File path
        human_readable: Return formatted string instead of bytes
        
    Returns:
        File size
    """
    size = Path(path).stat().st_size
    
    if not human_readable:
        return size
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def calculate_checksum(path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate file checksum.
    
    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hexadecimal checksum string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def find_files(directory: Union[str, Path], 
               pattern: str = '*',
               recursive: bool = True) -> List[Path]:
    """
    Find files matching a pattern in directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    dir_path = Path(directory)
    
    if recursive:
        return list(dir_path.rglob(pattern))
    else:
        return list(dir_path.glob(pattern))


def get_file_extension(path: Union[str, Path], include_dot: bool = False) -> str:
    """
    Get file extension from path.
    
    Args:
        path: File path
        include_dot: Whether to include leading dot
        
    Returns:
        File extension
    """
    ext = Path(path).suffix
    if not include_dot and ext.startswith('.'):
        ext = ext[1:]
    return ext


def change_file_extension(path: Union[str, Path], new_extension: str) -> Path:
    """
    Change file extension.
    
    Args:
        path: Original file path
        new_extension: New extension (with or without dot)
        
    Returns:
        Path with new extension
    """
    file_path = Path(path)
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    return file_path.with_suffix(new_extension)


def safe_filename(filename: str, replacement: str = '_') -> str:
    """
    Make a filename safe by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
        
    Returns:
        Safe filename
    """
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in invalid_chars:
        safe_name = safe_name.replace(char, replacement)
    
    safe_name = safe_name.strip('. ')
    
    return safe_name if safe_name else 'unnamed'


def get_unique_filename(directory: Union[str, Path], filename: str) -> Path:
    """
    Get a unique filename in directory by adding number suffix if needed.
    
    Args:
        directory: Target directory
        filename: Desired filename
        
    Returns:
        Unique file path
    """
    dir_path = Path(directory)
    file_path = dir_path / filename
    
    if not file_path.exists():
        return file_path
    
    name_without_ext = file_path.stem
    extension = file_path.suffix
    counter = 1
    
    while True:
        new_filename = f"{name_without_ext}_{counter}{extension}"
        new_path = dir_path / new_filename
        if not new_path.exists():
            return new_path
        counter += 1


def read_file_lines(path: Union[str, Path], 
                    strip_whitespace: bool = True,
                    skip_empty: bool = True) -> List[str]:
    """
    Read file lines with optional processing.
    
    Args:
        path: File path
        strip_whitespace: Whether to strip whitespace from lines
        skip_empty: Whether to skip empty lines
        
    Returns:
        List of lines
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if strip_whitespace:
        lines = [line.strip() for line in lines]
    
    if skip_empty:
        lines = [line for line in lines if line]
    
    return lines


def copy_file_metadata(source: Union[str, Path], dest: Union[str, Path]) -> None:
    """
    Copy file metadata (permissions, timestamps) from source to destination.
    
    Args:
        source: Source file path
        dest: Destination file path
    """
    src_stat = Path(source).stat()
    dest_path = Path(dest)
    
    os.utime(dest_path, (src_stat.st_atime, src_stat.st_mtime))
    os.chmod(dest_path, src_stat.st_mode)


# Data Conversion Utilities
# ==========================

def to_bool(value: Any) -> bool:
    """
    Convert various types to boolean.
    
    Args:
        value: Value to convert
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1', 'on', 'y')
    
    if isinstance(value, (int, float)):
        return value != 0
    
    return bool(value)


def to_int(value: Any, default: int = 0) -> int:
    """
    Convert value to integer with default fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value to float with default fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def bytes_to_string(data: bytes, encoding: str = 'utf-8') -> str:
    """
    Convert bytes to string with error handling.
    
    Args:
        data: Bytes to convert
        encoding: Character encoding
        
    Returns:
        Decoded string
    """
    try:
        return data.decode(encoding)
    except UnicodeDecodeError:
        return data.decode(encoding, errors='replace')


def string_to_bytes(text: str, encoding: str = 'utf-8') -> bytes:
    """
    Convert string to bytes.
    
    Args:
        text: String to convert
        encoding: Character encoding
        
    Returns:
        Encoded bytes
    """
    return text.encode(encoding)


def dict_to_query_string(params: Dict[str, Any]) -> str:
    """
    Convert dictionary to URL query string.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        URL-encoded query string
    """
    items = []
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            for item in value:
                items.append(f"{urllib.parse.quote(str(key))}={urllib.parse.quote(str(item))}")
        else:
            items.append(f"{urllib.parse.quote(str(key))}={urllib.parse.quote(str(value))}")
    return '&'.join(items)


def query_string_to_dict(query: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse URL query string to dictionary.
    
    Args:
        query: Query string
        
    Returns:
        Dictionary of parameters
    """
    result: Dict[str, Union[str, List[str]]] = {}
    
    if not query:
        return result
    
    if query.startswith('?'):
        query = query[1:]
    
    for item in query.split('&'):
        if '=' in item:
            key, value = item.split('=', 1)
            key = urllib.parse.unquote(key)
            value = urllib.parse.unquote(value)
            
            if key in result:
                if isinstance(result[key], list):
                    result[key].append(value)
                else:
                    result[key] = [result[key], value]
            else:
                result[key] = value
    
    return result


def flatten_dict(data: Dict[str, Any], 
                 parent_key: str = '',
                 separator: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        data: Dictionary to flatten
        parent_key: Parent key prefix
        separator: Key separator
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with separator in keys.
    
    Args:
        data: Flattened dictionary
        separator: Key separator
        
    Returns:
        Nested dictionary
    """
    result: Dict[str, Any] = {}
    
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


# Validation Utilities
# ====================

def is_valid_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL format
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url, re.IGNORECASE))


def is_valid_ipv4(ip: str) -> bool:
    """
    Validate IPv4 address.
    
    Args:
        ip: IP address to validate
        
    Returns:
        True if valid IPv4 address
    """
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    
    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False


def is_valid_phone(phone: str, country: str = 'US') -> bool:
    """
    Validate phone number format (basic validation).
    
    Args:
        phone: Phone number to validate
        country: Country code for validation rules
        
    Returns:
        True if valid phone format
    """
    digits = re.sub(r'[^\d]', '', phone)
    
    if country == 'US':
        return len(digits) == 10 or (len(digits) == 11 and digits[0] == '1')
    
    return 7 <= len(digits) <= 15


def is_valid_credit_card(number: str) -> bool:
    """
    Validate credit card number using Luhn algorithm.
    
    Args:
        number: Credit card number
        
    Returns:
        True if valid by Luhn algorithm
    """
    digits = re.sub(r'[^\d]', '', number)
    
    if not digits or not digits.isdigit():
        return False
    
    def luhn_checksum(card_number):
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        
        return checksum % 10
    
    return luhn_checksum(digits) == 0


def is_strong_password(password: str, 
                       min_length: int = 8,
                       require_uppercase: bool = True,
                       require_lowercase: bool = True,
                       require_digit: bool = True,
                       require_special: bool = True) -> bool:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        min_length: Minimum password length
        require_uppercase: Require uppercase letter
        require_lowercase: Require lowercase letter
        require_digit: Require digit
        require_special: Require special character
        
    Returns:
        True if password meets requirements
    """
    if len(password) < min_length:
        return False
    
    if require_uppercase and not any(c.isupper() for c in password):
        return False
    
    if require_lowercase and not any(c.islower() for c in password):
        return False
    
    if require_digit and not any(c.isdigit() for c in password):
        return False
    
    if require_special and not any(c in string.punctuation for c in password):
        return False
    
    return True


def validate_range(value: Union[int, float], 
                   min_value: Optional[Union[int, float]] = None,
                   max_value: Optional[Union[int, float]] = None) -> bool:
    """
    Validate if value is within range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if value in range
    """
    if min_value is not None and value < min_value:
        return False
    
    if max_value is not None and value > max_value:
        return False
    
    return True


# Common Operations
# =================

def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def deduplicate_list(items: List[T], preserve_order: bool = True) -> List[T]:
    """
    Remove duplicates from list.
    
    Args:
        items: List with duplicates
        preserve_order: Whether to preserve original order
        
    Returns:
        List without duplicates
    """
    if preserve_order:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        return list(set(items))


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_nested_value(data: Dict[str, Any], 
                     key_path: str, 
                     default: Any = None,
                     separator: str = '.') -> Any:
    """
    Get value from nested dictionary using dot notation.
    
    Args:
        data: Dictionary to search
        key_path: Dot-separated key path
        default: Default value if key not found
        separator: Key separator
        
    Returns:
        Value at key path or default
    """
    keys = key_path.split(separator)
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(data: Dict[str, Any], 
                     key_path: str, 
                     value: Any,
                     separator: str = '.') -> None:
    """
    Set value in nested dictionary using dot notation.
    
    Args:
        data: Dictionary to modify
        key_path: Dot-separated key path
        value: Value to set
        separator: Key separator
    """
    keys = key_path.split(separator)
    current = data
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def retry_operation(func: Callable, 
                    max_attempts: int = 3,
                    delay: float = 1.0,
                    backoff: float = 2.0) -> Any:
    """
    Retry an operation with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
        
    Returns:
        Function result
        
    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(current_delay)
                current_delay *= backoff
    
    raise last_exception


def memoize(func: Callable) -> Callable:
    """
    Memoization decorator for caching function results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


def clamp(value: Union[int, float], 
          min_value: Union[int, float], 
          max_value: Union[int, float]) -> Union[int, float]:
    """
    Clamp value between minimum and maximum.
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def percentage(part: Union[int, float], 
               whole: Union[int, float], 
               decimal_places: int = 2) -> float:
    """
    Calculate percentage.
    
    Args:
        part: Part value
        whole: Whole value
        decimal_places: Number of decimal places
        
    Returns:
        Percentage value
    """
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, decimal_places)


def generate_random_string(length: int, 
                          charset: str = string.ascii_letters + string.digits) -> str:
    """
    Generate random string.
    
    Args:
        length: Length of string
        charset: Characters to use
        
    Returns:
        Random string
    """
    return ''.join(random.choice(charset) for _ in range(length))


def