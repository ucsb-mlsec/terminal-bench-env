#!/usr/bin/env python3
"""
Main application module for the pyapp package.
Provides basic data processing and utility functions.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """A simple data processor for handling structured data."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.processed_count = 0
        logger.info(f"DataProcessor '{name}' initialized")
    
    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of records by adding metadata."""
        processed = []
        for record in records:
            processed_record = record.copy()
            processed_record['processed_at'] = datetime.now().isoformat()
            processed_record['processor'] = self.name
            processed.append(processed_record)
            self.processed_count += 1
        return processed
    
    def get_stats(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return {
            'processor_name': self.name,
            'total_processed': self.processed_count
        }


def calculate_sum(numbers: List[float]) -> float:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)


def calculate_average(numbers: List[float]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def main():
    """Main entry point for the application."""
    logger.info("Starting pyapp application")
    
    processor = DataProcessor("main_processor")
    
    sample_data = [
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
        {'id': 3, 'value': 300}
    ]
    
    processed = processor.process_records(sample_data)
    logger.info(f"Processed {len(processed)} records")
    
    values = [record['value'] for record in sample_data]
    total = calculate_sum(values)
    average = calculate_average(values)
    
    logger.info(f"Total: {total}, Average: {average}")
    logger.info(f"Stats: {processor.get_stats()}")
    
    return 0


if __name__ == "__main__":
    exit(main())