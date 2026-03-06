#!/usr/bin/env python
"""
Task definitions for the Celery-based data processing system.

NOTE: The task implementation logic is correct and does not need modification.
The timeout issues are purely configuration-related.
"""

from celery import Celery, Task
import time
import os

# Create Celery app instance
app = Celery('data_processor')

# Load configuration from celeryconfig.py
app.config_from_object('celeryconfig')


@app.task
def process_dataset(dataset_path, chunk_size=1024):
    """
    Process a dataset file with simulated computation time.
    
    This task simulates processing by calculating the file size and
    sleeping for a duration proportional to the file size.
    
    NOTE: This implementation is correct and does not need changes.
    The timeout failures are caused by configuration issues, not task logic.
    
    Args:
        dataset_path: Path to the dataset file to process
        chunk_size: Size of processing chunks (default: 1024)
    
    Returns:
        dict: Processing result with status, dataset path, size, and duration
    """
    # Get the file size
    size = os.path.getsize(dataset_path)
    
    # Calculate processing time based on file size
    # Approximately 3.6 seconds per MB
    file_size_mb = size / (1024 * 1024)
    processing_time = file_size_mb * 3.6
    
    # Simulate processing with sleep
    time.sleep(processing_time)
    
    # Return processing results
    return {
        'status': 'completed',
        'dataset': dataset_path,
        'size_mb': file_size_mb,
        'duration': processing_time
    }


@app.task
def analyze_results(result_data):
    """
    Analyze the results from dataset processing.
    
    Args:
        result_data: Dictionary containing processing results
    
    Returns:
        dict: Analysis summary
    """
    # Simple analysis of the processing results
    analysis = {
        'analyzed': True,
        'input_size': result_data.get('size_mb', 0),
        'processing_duration': result_data.get('duration', 0),
        'summary': f"Processed {result_data.get('dataset', 'unknown')} successfully"
    }
    
    return analysis