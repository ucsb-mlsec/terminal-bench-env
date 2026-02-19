#!/usr/bin/env python3
"""
Main Application Module for Data Processing System
This module provides comprehensive data processing, validation, and management capabilities
for handling large-scale data operations with multiple backend integrations.

Version: 2.4.1
Author: Development Team
License: MIT
"""

import os
import sys
import json
import logging
import hashlib
import time
import datetime
import argparse
import threading
import queue
import socket
import re
import pickle
import gzip
import shutil
import tempfile
import subprocess
import sqlite3
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, OrderedDict, Counter
from functools import wraps, lru_cache
from itertools import islice, chain
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings
import traceback
import signal
import copy
import base64
import zlib
import math
import random
import string
import unicodedata
import mimetypes
import fnmatch
import glob
import configparser
import urllib.parse
import urllib.request
from urllib.error import URLError, HTTPError

# Third-party imports simulation
try:
    import requests
except ImportError:
    requests = None

try:
    import numpy as np
except ImportError:
    np = None

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enumeration for processing status states"""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()
    PAUSED = auto()


class DataType(Enum):
    """Enumeration for supported data types"""
    TEXT = auto()
    BINARY = auto()
    JSON = auto()
    XML = auto()
    CSV = auto()
    PARQUET = auto()
    AVRO = auto()


@dataclass
class ProcessingConfig:
    """Configuration dataclass for processing operations"""
    batch_size: int = 1000
    max_workers: int = 4
    timeout: int = 300
    retry_attempts: int = 3
    enable_compression: bool = True
    compression_level: int = 6
    validate_checksums: bool = True
    preserve_metadata: bool = True
    encryption_enabled: bool = False
    log_verbosity: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration from dictionary"""
        return cls(**data)


@dataclass
class DataRecord:
    """Data record structure for processing"""
    record_id: str
    timestamp: datetime.datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def __hash__(self):
        return hash(self.record_id)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


class ConnectionError(Exception):
    """Custom exception for connection errors"""
    pass


class DataValidator:
    """Comprehensive data validation utility class"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.validation_rules = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_rule(self, field_name: str, rule_func: callable, error_message: str):
        """Add a validation rule for a specific field"""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        self.validation_rules[field_name].append((rule_func, error_message))
        self.logger.debug(f"Added validation rule for field: {field_name}")
    
    def validate_email(self, email: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        pattern = r'^\+?1?\d{9,15}$'
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)
        return re.match(pattern, cleaned) is not None
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False
    
    def validate_date(self, date_string: str, format_string: str = "%Y-%m-%d") -> bool:
        """Validate date string against format"""
        try:
            datetime.datetime.strptime(date_string, format_string)
            return True
        except ValueError:
            return False
    
    def validate_json(self, json_string: str) -> bool:
        """Validate JSON string"""
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False
    
    def validate_range(self, value: Union[int, float], min_val: Union[int, float], 
                      max_val: Union[int, float]) -> bool:
        """Validate numeric value within range"""
        return min_val <= value <= max_val
    
    def validate_length(self, value: str, min_len: int, max_len: int) -> bool:
        """Validate string length"""
        return min_len <= len(value) <= max_len
    
    def validate_pattern(self, value: str, pattern: str) -> bool:
        """Validate string against regex pattern"""
        return re.match(pattern, value) is not None
    
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate entire record against all rules"""
        errors = []
        for field_name, rules in self.validation_rules.items():
            if field_name not in record:
                errors.append(f"Missing required field: {field_name}")
                continue
            
            field_value = record[field_name]
            for rule_func, error_message in rules:
                try:
                    if not rule_func(field_value):
                        errors.append(f"{field_name}: {error_message}")
                except Exception as e:
                    errors.append(f"{field_name}: Validation error - {str(e)}")
        
        return len(errors) == 0, errors
    
    def batch_validate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate multiple records and return summary"""
        results = {
            'total': len(records),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        for idx, record in enumerate(records):
            is_valid, errors = self.validate_record(record)
            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['errors'].append({
                    'record_index': idx,
                    'errors': errors
                })
        
        return results


class DataTransformer:
    """Data transformation and conversion utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.transformation_cache = {}
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing special characters and standardizing"""
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip().lower()
    
    def convert_case(self, text: str, case_type: str) -> str:
        """Convert text case"""
        case_type = case_type.lower()
        if case_type == 'upper':
            return text.upper()
        elif case_type == 'lower':
            return text.lower()
        elif case_type == 'title':
            return text.title()
        elif case_type == 'camel':
            words = text.split('_')
            return words[0].lower() + ''.join(w.capitalize() for w in words[1:])
        elif case_type == 'snake':
            return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
        return text
    
    def parse_csv_line(self, line: str, delimiter: str = ',') -> List[str]:
        """Parse CSV line handling quoted fields"""
        return list(csv.reader([line], delimiter=delimiter))[0]
    
    def dict_to_xml(self, data: Dict[str, Any], root_tag: str = 'root') -> str:
        """Convert dictionary to XML string"""
        def build_element(parent, key, value):
            if isinstance(value, dict):
                element = ET.SubElement(parent, key)
                for k, v in value.items():
                    build_element(element, k, v)
            elif isinstance(value, list):
                for item in value:
                    build_element(parent, key, item)
            else:
                element = ET.SubElement(parent, key)
                element.text = str(value)
        
        root = ET.Element(root_tag)
        for key, value in data.items():
            build_element(root, key, value)
        
        return ET.tostring(root, encoding='unicode')
    
    def xml_to_dict(self, xml_string: str) -> Dict[str, Any]:
        """Convert XML string to dictionary"""
        def parse_element(element):
            result = {}
            if len(element) == 0:
                return element.text
            
            for child in element:
                child_data = parse_element(child)
                if child.tag in result:
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data
            
            return result
        
        root = ET.fromstring(xml_string)
        return {root.tag: parse_element(root)}
    
    def flatten_dict(self, data: Dict[str, Any], parent_key: str = '', 
                     separator: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self.flatten_dict(value, new_key, separator).items())
            else:
                items.append((new_key, value))
        return dict(items)
    
    def unflatten_dict(self, data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """Unflatten dictionary with dot notation"""
        result = {}
        for key, value in data.items():
            parts = key.split(separator)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result
    
    def merge_dicts(self, *dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge multiple dictionaries"""
        result = {}
        for dictionary in dicts:
            for key, value in dictionary.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self.merge_dicts(result[key], value)
                else:
                    result[key] = value
        return result
    
    def transform_keys(self, data: Dict[str, Any], key_func: callable) -> Dict[str, Any]:
        """Transform all keys in dictionary using provided function"""
        if not isinstance(data, dict):
            return data
        
        result = {}
        for key, value in data.items():
            new_key = key_func(key)
            if isinstance(value, dict):
                result[new_key] = self.transform_keys(value, key_func)
            elif isinstance(value, list):
                result[new_key] = [
                    self.transform_keys(item, key_func) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[new_key] = value
        return result


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute SELECT query and return results"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            self.logger.debug(f"Query executed: {query}")
            return results
        except sqlite3.Error as e:
            self.logger.error(f"Query execution error: {e}")
            raise
    
    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            affected = cursor.rowcount
            self.logger.debug(f"Update executed, {affected} rows affected")
            return affected
        except sqlite3.Error as e:
            self.connection.rollback()
            self.logger.error(f"Update execution error: {e}")
            raise
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute query with multiple parameter sets"""
        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, params_list)
            self.connection.commit()
            affected = cursor.rowcount
            self.logger.debug(f"Batch update executed, {affected} rows affected")
            return affected
        except sqlite3.Error as e:
            self.connection.rollback()
            self.logger.error(f"Batch execution error: {e}")
            raise
    
    def create_table(self, table_name: str, schema: Dict[str, str]):
        """Create table with given schema"""
        columns = ', '.join([f"{col} {dtype}" for col, dtype in schema.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.execute_update(query)
        self.logger.info(f"Table created: {table_name}")
    
    def insert_record(self, table_name: str, record: Dict[str, Any]) -> int:
        """Insert single record into table"""
        columns = ', '.join(record.keys())
        placeholders = ', '.join(['?' for _ in record])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        return self.execute_update(query, tuple(record.values()))
    
    def bulk_insert(self, table_name: str, records: List[Dict[str, Any]]) -> int:
        """Bulk insert records"""
        if not records:
            return 0
        
        columns = ', '.join(records[0].keys())
        placeholders = ', '.join(['?' for _ in records[0]])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        params_list = [tuple(record.values()) for record in records]
        return self.execute_many(query, params_list)
    
    def update_record(self, table_name: str, record: Dict[str, Any], 
                     condition: str, condition_params: Tuple = ()) -> int:
        """Update record with condition"""
        set_clause = ', '.join([f"{col} = ?" for col in record.keys()])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
        params = tuple(record.values()) + condition_params
        return self.execute_update(query, params)
    
    def delete_records(self, table_name: str, condition: str, 
                      params: Tuple = ()) -> int:
        """Delete records matching condition"""
        query = f"DELETE FROM {table_name} WHERE {condition}"
        return self.execute_update(query, params)
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information"""
        query = f"PRAGMA table_info({table_name})"
        results = self.execute_query(query)
        return [dict(row) for row in results]
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        results = self.execute_query(query, (table_name,))
        return len(results) > 0


class FileProcessor:
    """File processing and management utilities"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_checksum(self, file_path: str, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        hash_obj = hashlib.new(algorithm)
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except IOError as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            raise
    
    def compress_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Compress file using gzip"""
        if output_path is None:
            output_path = f"{input_path}.gz"
        
        try:
            with open(input_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb', 
                              compresslevel=self.config.compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            self.logger.info(f"File compressed: {output_path}")
            return output_path
        except IOError as e:
            self.logger.error(f"Error compressing file: {e}")
            raise
    
    def decompress_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Decompress gzip file"""
        if output_path is None:
            output_path = input_path.rsplit('.gz', 1)[0]
        
        try:
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            self.logger.info(f"File decompressed: {output_path}")
            return output_path
        except IOError as e:
            self.logger.error(f"Error decompressing file: {e}")
            raise
    
    def split_file(self, file_path: str, chunk_size: int) -> List[str]:
        """Split file into chunks"""
        chunk_files = []
        base_name = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        
        try:
            with open(file_path, 'rb') as f:
                chunk_num = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunk_file = os.path.join(dir_name, f"{base_name}.part{chunk_num}")
                    with open(chunk_file, 'wb') as chunk_f:
                        chunk_f.write(chunk)
                    
                    chunk_files.append(chunk_file)
                    chunk_num += 1
            
            self.logger.info(f"File split into {len(chunk_files)} chunks")
            return chunk_files
        except IOError as e:
            self.logger.error(f"Error splitting file: {e}")
            raise
    
    def merge_files(self, chunk_files: List[str], output_path: str):
        """Merge file chunks"""
        try:
            with open(output_path, 'wb') as output_f:
                for chunk_file in sorted(chunk_files):
                    with open(chunk_file, 'rb') as chunk_f:
                        shutil.copyfileobj(chunk_f, output_f)
            
            self.logger.info(f"Files merged to: {output_path}")
        except IOError as e:
            self.logger.error(f"Error merging files: {e}")
            raise
    
    def read_file_lines(self, file_path: str, encoding: str = 'utf-8') -> List[str]:
        """Read all lines from file"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.readlines()
        except IOError as e:
            self.logger.error(f"Error reading file: {e}")
            raise
    
    def write_file_lines(self, file_path: str, lines: List[str], 
                        encoding: str = 'utf-8'):
        """Write lines to file"""
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.writelines(lines)
        except IOError as e:
            self.logger.error(f"Error writing file: {e}")
            raise
    
    def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read JSON from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"Error reading JSON file: {e}")
            raise
    
    def write_json_file(self, file_path: str, data: Dict[str, Any], 
                       indent: int = 2):
        """Write JSON to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
        except IOError as e:
            self.logger.error(f"Error writing JSON file: {e}")
            raise
    
    def copy_file_with_metadata(self, source: str, destination: str):
        """Copy file preserving metadata"""
        try:
            shutil.copy2(source, destination)
            self.logger.debug(f"File copied with metadata: {source} -> {destination}")
        except IOError as e:
            self.logger.error(f"Error copying file: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed file information"""
        try:
            stats = os.stat(file_path)
            return {
                'path': file_path,
                'size': stats.st_size,
                'created': datetime.datetime.fromtimestamp(stats.st_ctime),
                'modified': datetime.datetime.fromtimestamp(stats.st_mtime),
                'accessed': datetime.datetime.fromtimestamp(stats.st_atime),
                'permissions': oct(stats.st_mode)[-3:],
                'is_file': os.path.isfile(file_path),
                'is_dir': os.path.isdir(file_path),
                'is_link': os.path.islink(file_path)
            }
        except OSError as e:
            self.logger.error(f"Error getting file info: {e}")
            raise


class APIClient:
    """Generic API client for HTTP operations"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session_headers = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def set_header(self, key: str, value: str):
        """Set session header"""
        self.session_headers[key] = value
    
    def set_auth_token(self, token: str):
        """Set authorization token"""
        self.set_header('Authorization', f'Bearer {token}')
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.session_headers, **kwargs.get('headers', {})}
        
        if 'headers' in kwargs:
            del kwargs['headers']
        
        try:
            if requests:
                response = requests.request(
                    method, url, headers=headers, timeout=self.timeout, **kwargs
                )
                response.raise_for_status()
                return response.json() if response.content else {}
            else:
                req = urllib.request.Request(url, method=method.upper(), headers=headers)
                if 'data' in kwargs:
                    req.data = json.dumps(kwargs['data']).encode('utf-8')
                
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    content = response.read().decode('utf-8')
                    return json.loads(content) if content else {}
        except Exception as e:
            self.logger.error(f"API request error: {e}")
            raise
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make POST request"""
        return self._make_request('POST', endpoint, json=data)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make PUT request"""
        return self._make_request('PUT', endpoint, json=data)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request"""
        return self._make_request('DELETE', endpoint)
    
    def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make PATCH request"""
        return self._make_request('PATCH', endpoint, json=data)


class CacheManager:
    """Cache management system"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            if time.time() - self.timestamps[key] > self.ttl:
                self.delete(key)
                self.misses += 1
                return None
            
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def delete(self, key: str):
        """Delete value from cache"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'ttl': self.ttl
        }


class DataProcessor:
    """Main data processing engine"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.validator = DataValidator(self.config)
        self.transformer = DataTransformer()
        self.file_processor = FileProcessor(self.config)
        self.cache = CacheManager()
        self.logger