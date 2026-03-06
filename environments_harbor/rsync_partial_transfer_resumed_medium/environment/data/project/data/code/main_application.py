#!/usr/bin/env python3
"""
Main Application for Data Processing and Analytics Platform
This module provides comprehensive data processing capabilities including
ETL operations, API integrations, database management, and analytics.
"""

import os
import sys
import json
import time
import logging
import hashlib
import sqlite3
import asyncio
import threading
import multiprocessing
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps, lru_cache
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import csv
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from enum import Enum, auto
import re
import uuid
import gzip
import shutil
from io import StringIO, BytesIO

# Third-party imports (would normally be installed via pip)
try:
    import requests
    import pandas as pd
    import numpy as np
except ImportError as e:
    logging.warning(f"Optional dependency not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enumeration for data processing status"""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()


class DataType(Enum):
    """Enumeration for supported data types"""
    JSON = auto()
    CSV = auto()
    XML = auto()
    BINARY = auto()
    TEXT = auto()
    PARQUET = auto()


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations"""
    batch_size: int = 1000
    max_retries: int = 3
    timeout: int = 300
    enable_caching: bool = True
    cache_ttl: int = 3600
    parallel_workers: int = 4
    compression_enabled: bool = True
    validation_strict: bool = True
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class DataRecord:
    """Represents a single data record"""
    record_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation"""
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'metadata': self.metadata,
            'status': self.status.name,
            'retry_count': self.retry_count,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataRecord':
        """Create record from dictionary"""
        return cls(
            record_id=data['record_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            metadata=data.get('metadata', {}),
            status=ProcessingStatus[data.get('status', 'PENDING')],
            retry_count=data.get('retry_count', 0),
            error_message=data.get('error_message')
        )


class ConfigurationManager:
    """Manages application configuration and settings"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = {}
        self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self.config = self._get_default_config()
                self.save_config()
                logger.info("Default configuration created")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'dataprocessing',
                'user': 'admin',
                'pool_size': 10
            },
            'api': {
                'base_url': 'https://api.example.com',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit': 100
            },
            'processing': {
                'batch_size': 1000,
                'parallel_workers': 4,
                'checkpoint_enabled': True
            },
            'storage': {
                'data_dir': '/var/data',
                'backup_dir': '/var/backup',
                'retention_days': 30
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save_config()


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path: str = "application.db"):
        self.db_path = db_path
        self.connection = None
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize database schema"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            cursor = self.connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT,
                    status TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_records INTEGER DEFAULT 0,
                    processed_records INTEGER DEFAULT 0,
                    failed_records INTEGER DEFAULT 0,
                    started_at TEXT,
                    completed_at TEXT,
                    config TEXT,
                    result TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    user TEXT,
                    action TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    details TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_record_id ON data_records(record_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON data_records(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_id ON processing_jobs(job_id)')
            
            self.connection.commit()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def insert_record(self, record: DataRecord) -> bool:
        """Insert a data record"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO data_records 
                (record_id, timestamp, data, metadata, status, retry_count, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.record_id,
                record.timestamp.isoformat(),
                json.dumps(record.data),
                json.dumps(record.metadata),
                record.status.name,
                record.retry_count,
                record.error_message
            ))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting record: {e}")
            return False
    
    def get_record(self, record_id: str) -> Optional[DataRecord]:
        """Retrieve a data record"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM data_records WHERE record_id = ?', (record_id,))
            row = cursor.fetchone()
            if row:
                return DataRecord(
                    record_id=row['record_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    data=json.loads(row['data']),
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    status=ProcessingStatus[row['status']],
                    retry_count=row['retry_count'],
                    error_message=row['error_message']
                )
            return None
        except Exception as e:
            logger.error(f"Error retrieving record: {e}")
            return None
    
    def update_record_status(self, record_id: str, status: ProcessingStatus, 
                           error_message: Optional[str] = None) -> bool:
        """Update record status"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE data_records 
                SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE record_id = ?
            ''', (status.name, error_message, record_id))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating record status: {e}")
            return False
    
    def get_records_by_status(self, status: ProcessingStatus, limit: int = 100) -> List[DataRecord]:
        """Get records by status"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM data_records WHERE status = ? LIMIT ?
            ''', (status.name, limit))
            rows = cursor.fetchall()
            return [DataRecord(
                record_id=row['record_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                data=json.loads(row['data']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                status=ProcessingStatus[row['status']],
                retry_count=row['retry_count'],
                error_message=row['error_message']
            ) for row in rows]
        except Exception as e:
            logger.error(f"Error getting records by status: {e}")
            return []
    
    def create_processing_job(self, job_id: str, job_type: str, config: Dict[str, Any]) -> bool:
        """Create a new processing job"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO processing_jobs (job_id, job_type, status, config, started_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (job_id, job_type, ProcessingStatus.IN_PROGRESS.name, 
                  json.dumps(config), datetime.now().isoformat()))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating processing job: {e}")
            return False
    
    def update_job_progress(self, job_id: str, processed: int, failed: int) -> bool:
        """Update job progress"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE processing_jobs 
                SET processed_records = ?, failed_records = ?
                WHERE job_id = ?
            ''', (processed, failed, job_id))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating job progress: {e}")
            return False
    
    def complete_job(self, job_id: str, status: ProcessingStatus, result: Dict[str, Any]) -> bool:
        """Mark job as completed"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE processing_jobs 
                SET status = ?, completed_at = ?, result = ?
                WHERE job_id = ?
            ''', (status.name, datetime.now().isoformat(), json.dumps(result), job_id))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error completing job: {e}")
            return False
    
    def log_audit_event(self, action: str, user: str = "system", 
                       entity_type: Optional[str] = None, 
                       entity_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None) -> bool:
        """Log an audit event"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO audit_log (user, action, entity_type, entity_id, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (user, action, entity_type, entity_id, 
                  json.dumps(details) if details else None))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


class CacheManager:
    """Manages caching operations"""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        self.memory_cache = OrderedDict()
        self.max_memory_items = 1000
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Check memory cache first
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end (LRU)
                self.memory_cache.move_to_end(key)
                return value
            else:
                del self.memory_cache[key]
        
        # Check file cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if time.time() - cached_data['timestamp'] < self.ttl:
                        value = cached_data['value']
                        # Store in memory cache
                        self._set_memory_cache(key, value)
                        return value
                    else:
                        cache_path.unlink()
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Store in memory cache
        self._set_memory_cache(key, value)
        
        # Store in file cache
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def _set_memory_cache(self, key: str, value: Any):
        """Set value in memory cache with LRU eviction"""
        self.memory_cache[key] = (value, time.time())
        if len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False)
    
    def delete(self, key: str):
        """Delete value from cache"""
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self):
        """Clear all cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        logger.info("Cache cleared")
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        
        # Clean memory cache
        expired_keys = [k for k, (v, t) in self.memory_cache.items() 
                       if current_time - t >= self.ttl]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Clean file cache
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if current_time - cached_data['timestamp'] >= self.ttl:
                        cache_file.unlink()
            except Exception as e:
                logger.error(f"Error cleaning cache file: {e}")


class DataValidator:
    """Validates data records"""
    
    @staticmethod
    def validate_record(record: DataRecord, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate record against schema"""
        errors = []
        
        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in record.data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        field_types = schema.get('properties', {})
        for field, field_schema in field_types.items():
            if field in record.data:
                expected_type = field_schema.get('type')
                actual_value = record.data[field]
                
                if not DataValidator._check_type(actual_value, expected_type):
                    errors.append(f"Invalid type for field {field}: expected {expected_type}")
                
                # Check constraints
                if 'min' in field_schema and actual_value < field_schema['min']:
                    errors.append(f"Field {field} below minimum value")
                if 'max' in field_schema and actual_value > field_schema['max']:
                    errors.append(f"Field {field} above maximum value")
                if 'pattern' in field_schema:
                    if not re.match(field_schema['pattern'], str(actual_value)):
                        errors.append(f"Field {field} does not match pattern")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        expected = type_mapping.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number"""
        pattern = r'^\+?1?\d{9,15}$'
        return re.match(pattern, phone.replace('-', '').replace(' ', '')) is not None
    
    @staticmethod
    def validate_date(date_str: str, format: str = "%Y-%m-%d") -> bool:
        """Validate date string"""
        try:
            datetime.strptime(date_str, format)
            return True
        except:
            return False


class DataTransformer:
    """Transforms data records"""
    
    @staticmethod
    def transform_record(record: DataRecord, transformations: List[Dict[str, Any]]) -> DataRecord:
        """Apply transformations to record"""
        transformed_data = record.data.copy()
        
        for transformation in transformations:
            trans_type = transformation.get('type')
            field = transformation.get('field')
            
            if trans_type == 'rename':
                new_name = transformation.get('new_name')
                if field in transformed_data:
                    transformed_data[new_name] = transformed_data.pop(field)
            
            elif trans_type == 'convert':
                target_type = transformation.get('target_type')
                if field in transformed_data:
                    transformed_data[field] = DataTransformer._convert_type(
                        transformed_data[field], target_type
                    )
            
            elif trans_type == 'compute':
                expression = transformation.get('expression')
                transformed_data[field] = DataTransformer._evaluate_expression(
                    expression, transformed_data
                )
            
            elif trans_type == 'filter':
                condition = transformation.get('condition')
                if not DataTransformer._evaluate_condition(condition, transformed_data):
                    record.status = ProcessingStatus.FAILED
                    record.error_message = "Failed filter condition"
                    return record
            
            elif trans_type == 'split':
                delimiter = transformation.get('delimiter', ',')
                if field in transformed_data:
                    transformed_data[field] = str(transformed_data[field]).split(delimiter)
            
            elif trans_type == 'join':
                delimiter = transformation.get('delimiter', ',')
                fields = transformation.get('fields', [])
                values = [str(transformed_data.get(f, '')) for f in fields]
                transformed_data[field] = delimiter.join(values)
            
            elif trans_type == 'uppercase':
                if field in transformed_data:
                    transformed_data[field] = str(transformed_data[field]).upper()
            
            elif trans_type == 'lowercase':
                if field in transformed_data:
                    transformed_data[field] = str(transformed_data[field]).lower()
            
            elif trans_type == 'trim':
                if field in transformed_data:
                    transformed_data[field] = str(transformed_data[field]).strip()
            
            elif trans_type == 'default':
                default_value = transformation.get('default_value')
                if field not in transformed_data or transformed_data[field] is None:
                    transformed_data[field] = default_value
        
        record.data = transformed_data
        return record
    
    @staticmethod
    def _convert_type(value: Any, target_type: str) -> Any:
        """Convert value to target type"""
        try:
            if target_type == 'string':
                return str(value)
            elif target_type == 'integer':
                return int(value)
            elif target_type == 'float':
                return float(value)
            elif target_type == 'boolean':
                return bool(value)
            elif target_type == 'date':
                return datetime.fromisoformat(str(value))
            else:
                return value
        except:
            return value
    
    @staticmethod
    def _evaluate_expression(expression: str, data: Dict[str, Any]) -> Any:
        """Evaluate expression with data context"""
        # Simple expression evaluation - in production use a proper expression parser
        try:
            # Replace field references with values
            for key, value in data.items():
                expression = expression.replace(f'${key}', str(value))
            return eval(expression)
        except:
            return None
    
    @staticmethod
    def _evaluate_condition(condition: str, data: Dict[str, Any]) -> bool:
        """Evaluate condition with data context"""
        try:
            for key, value in data.items():
                condition = condition.replace(f'${key}', repr(value))
            return bool(eval(condition))
        except:
            return False


class APIClient:
    """Client for API interactions"""
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session"""
        try:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'DataProcessingApp/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            })
        except:
            logger.warning("requests library not available")
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make GET request"""
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if self.session:
                    response = self.session.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
                else:
                    # Fallback simulation
                    return {"status": "simulated", "data": {}}
            except Exception as e:
                logger.error(f"GET request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make POST request"""
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if self.session:
                    response = self.session.post(url, json=data, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
                else:
                    return {"status": "simulated", "data": {}}
            except Exception as e:
                logger.error(f"POST request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
    
    def put(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make PUT request"""
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if self.session:
                    response = self.session.put(url, json=data, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
                else:
                    return {"status": "simulated", "data": {}}
            except Exception as e:
                logger.error(f"PUT request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
    
    def delete(self, endpoint: str) -> bool:
        """Make DELETE request"""
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if self.session:
                    response = self.session.delete(url, timeout=self.timeout)
                    response.raise_for_status()
                    return True
                else:
                    return True
            except Exception as e:
                logger.error(f"DELETE request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False


class DataProcessor:
    """Main data processing engine"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager(ttl=config.cache_ttl)
        self.config_manager = ConfigurationManager()
        self.api_client = APIClient(
            base_url=self.config_manager.get('api.base_url', 'http://localhost'),
            timeout=self.config_manager.get('api.timeout', 30)
        )
        self.processing_queue = []
        self.results = []
        
    def process_batch(self, records: List[DataRecord]) -> Dict[str, Any]:
        """Process a batch of records"""
        job_id = str(uuid.uuid4())
        self.db_manager.create_processing_job(job_id, 'batch_processing', 
                                             self.config.__dict__)
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        for i, record in enumerate(records):
            try:
                # Validate record
                is_valid, errors = DataValidator.validate_record(
                    record, self._get_validation_schema()
                )
                
                if not is_valid:
                    record.status = Processing