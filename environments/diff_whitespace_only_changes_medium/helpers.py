#!/usr/bin/env python3

import json


def process_data(data):
    """Process input data and return transformed result."""
    if not data:
        return []
    
    processed = []
    for item in data:
        if isinstance(item, str):
            processed.append(item.upper())  
        else:
            processed.append(item)
    
    return processed


def load_config():
    """Load and return configuration settings."""
    config = {
        'database': 'postgresql',
        'host': 'localhost',
        'port': 5432,   
        'debug': True
    }
    return config



class DataProcessor:
    """A class for processing data with configuration."""
    
    def __init__(self, config=None):
        self.config = config or load_config()
        self.processed_count = 0
    
    def process(self, data):
        """Process data using the configured settings."""
        result = process_data(data)
        self.processed_count += len(result)  
        return result