#!/usr/bin/env python3
"""
Data Processor Module
Handles data transformation and aggregation operations
"""

import json
import logging
from collections import defaultdict, OrderedDict
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processing class for handling various data transformations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data processor
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.stats = defaultdict(int)
        self.processed_count = 0
        logger.info("DataProcessor initialized")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for incoming data
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed data dictionary
        """
        logger.debug(f"Processing data: {data.get('id', 'unknown')}")
        
        # Line 45: Normal dictionary allocation for processing
        result = {
            'id': data.get('id'),
            'timestamp': data.get('timestamp'),
            'status': 'processed',
            'metadata': self._extract_metadata(data)
        }
        
        # Transform the data
        transformed = self.transform(result)
        
        # Update statistics
        self.stats['processed'] += 1
        self.processed_count += 1
        
        return transformed
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data according to configured rules
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        transformed = data.copy()
        
        # Apply transformations
        if 'metadata' in transformed:
            transformed['metadata'] = self._normalize_metadata(transformed['metadata'])
        
        # Add processing markers
        transformed['processed_by'] = 'DataProcessor'
        transformed['version'] = self.config.get('version', '1.0')
        
        return transformed
    
    def aggregate(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple data items
        
        Args:
            data_list: List of data dictionaries
            
        Returns:
            Aggregated result
        """
        aggregated = {
            'count': len(data_list),
            'items': [],
            'summary': defaultdict(int)
        }
        
        for item in data_list:
            processed = self.process(item)
            aggregated['items'].append(processed)
            
            # Update summary statistics
            if 'status' in processed:
                aggregated['summary'][processed['status']] += 1
        
        return dict(aggregated)
    
    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from data
        
        Args:
            data: Source data
            
        Returns:
            Extracted metadata
        """
        metadata = {}
        
        for key in ['source', 'type', 'category', 'tags']:
            if key in data:
                metadata[key] = data[key]
        
        return metadata
    
    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metadata fields
        
        Args:
            metadata: Raw metadata
            
        Returns:
            Normalized metadata
        """
        normalized = OrderedDict()
        
        for key, value in sorted(metadata.items()):
            if isinstance(value, str):
                normalized[key] = value.lower().strip()
            else:
                normalized[key] = value
        
        return dict(normalized)
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get processing statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_processed': self.processed_count,
            **dict(self.stats)
        }
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self.stats.clear()
        self.processed_count = 0
        logger.info("Statistics reset")


def create_processor(config_path: Optional[str] = None) -> DataProcessor:
    """
    Factory function to create a DataProcessor instance
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured DataProcessor instance
    """
    config = {}
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    return DataProcessor(config)