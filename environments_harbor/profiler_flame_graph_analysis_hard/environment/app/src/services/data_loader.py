#!/usr/bin/env python3
"""
Data Loader Service Module

This module provides functionality for loading and parsing data from various sources
including JSON, CSV, and text files.
"""

import os
import json
import csv
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class for loading and processing data from multiple file formats.
    
    Supports JSON, CSV, and text file formats with validation and error handling.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            base_path: Optional base directory path for file loading
        """
        self.base_path = base_path or os.getcwd()
        self.loaded_data = []
        self.validation_errors = []
        logger.info(f"DataLoader initialized with base path: {self.base_path}")
    
    def load_file(self, filename: str, file_type: Optional[str] = None) -> bool:
        """
        Load a file and parse its contents.
        
        Args:
            filename: Name or path of the file to load
            file_type: Optional explicit file type ('json', 'csv', 'txt')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.base_path, filename)
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Determine file type
            if file_type is None:
                file_type = self._detect_file_type(filename)
            
            logger.info(f"Loading file: {filename} as type: {file_type}")
            
            if file_type == 'json':
                return self._load_json(file_path)
            elif file_type == 'csv':
                return self._load_csv(file_path)
            elif file_type == 'txt':
                return self._load_text(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading file {filename}: {str(e)}")
            return False
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from extension."""
        ext = os.path.splitext(filename)[1].lower()
        
        type_mapping = {
            '.json': 'json',
            '.csv': 'csv',
            '.txt': 'txt',
            '.log': 'txt'
        }
        
        return type_mapping.get(ext, 'txt')
    
    def _load_json(self, file_path: str) -> bool:
        """Load and parse JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.loaded_data.append({
                    'source': file_path,
                    'type': 'json',
                    'data': data
                })
            logger.info(f"Successfully loaded JSON from {file_path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {str(e)}")
            return False
    
    def _load_csv(self, file_path: str) -> bool:
        """Load and parse CSV file."""
        try:
            rows = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            
            self.loaded_data.append({
                'source': file_path,
                'type': 'csv',
                'data': rows
            })
            logger.info(f"Successfully loaded {len(rows)} rows from {file_path}")
            return True
        except Exception as e:
            logger.error(f"CSV parse error in {file_path}: {str(e)}")
            return False
    
    def _load_text(self, file_path: str) -> bool:
        """Load text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.loaded_data.append({
                    'source': file_path,
                    'type': 'text',
                    'data': content
                })
            logger.info(f"Successfully loaded text from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Text file read error in {file_path}: {str(e)}")
            return False
    
    def parse_data(self, data_item: Dict[str, Any]) -> List[str]:
        """
        Parse a data item into structured strings.
        
        Args:
            data_item: Dictionary containing data to parse
            
        Returns:
            List of parsed string representations
        """
        parsed_items = []
        
        try:
            data_type = data_item.get('type', 'unknown')
            data = data_item.get('data')
            
            if data_type == 'json':
                parsed_items = self._parse_json_data(data)
            elif data_type == 'csv':
                parsed_items = self._parse_csv_data(data)
            elif data_type == 'text':
                parsed_items = self._parse_text_data(data)
            
            logger.debug(f"Parsed {len(parsed_items)} items from {data_type} data")
            
        except Exception as e:
            logger.error(f"Error parsing data: {str(e)}")
        
        return parsed_items
    
    def _parse_json_data(self, data: Any) -> List[str]:
        """Parse JSON data into strings."""
        results = []
        if isinstance(data, dict):
            for key, value in data.items():
                results.append(f"{key}={str(value)}")
        elif isinstance(data, list):
            for item in data:
                results.append(str(item))
        else:
            results.append(str(data))
        return results
    
    def _parse_csv_data(self, data: List[Dict]) -> List[str]:
        """Parse CSV data into strings."""
        results = []
        for row in data:
            row_str = ", ".join([f"{k}:{v}" for k, v in row.items()])
            results.append(row_str)
        return results
    
    def _parse_text_data(self, data: str) -> List[str]:
        """Parse text data into lines."""
        return [line.strip() for line in data.split('\n') if line.strip()]
    
    def validate(self, rules: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate loaded data against optional rules.
        
        Args:
            rules: Dictionary containing validation rules
            
        Returns:
            True if all data is valid, False otherwise
        """
        self.validation_errors = []
        
        if not self.loaded_data:
            logger.warning("No data loaded for validation")
            return True
        
        for idx, data_item in enumerate(self.loaded_data):
            if not self._validate_item(data_item, rules):
                self.validation_errors.append(f"Item {idx} failed validation")
        
        if self.validation_errors:
            logger.warning(f"Validation found {len(self.validation_errors)} errors")
            return False
        
        logger.info("All data validated successfully")
        return True
    
    def _validate_item(self, item: Dict[str, Any], rules: Optional[Dict[str, Any]]) -> bool:
        """Validate a single data item."""
        if rules is None:
            return True
        
        # Basic validation - check required fields
        required_fields = rules.get('required_fields', [])
        data = item.get('data', {})
        
        if isinstance(data, dict):
            for field in required_fields:
                if field not in data:
                    return False
        
        return True
    
    def get_loaded_count(self) -> int:
        """Return the number of loaded data items."""
        return len(self.loaded_data)
    
    def clear_data(self):
        """Clear all loaded data."""
        self.loaded_data = []
        self.validation_errors = []
        logger.info("Cleared all loaded data")
    
    def export_summary(self) -> Dict[str, Any]:
        """Export a summary of loaded data."""
        return {
            'total_items': len(self.loaded_data),
            'types': [item['type'] for item in self.loaded_data],
            'sources': [item['source'] for item in self.loaded_data],
            'validation_errors': len(self.validation_errors)
        }


def load_multiple_files(file_list: List[str], base_path: Optional[str] = None) -> DataLoader:
    """
    Convenience function to load multiple files at once.
    
    Args:
        file_list: List of filenames to load
        base_path: Optional base directory path
        
    Returns:
        DataLoader instance with loaded data
    """
    loader = DataLoader(base_path)
    
    for filename in file_list:
        loader.load_file(filename)
    
    return loader