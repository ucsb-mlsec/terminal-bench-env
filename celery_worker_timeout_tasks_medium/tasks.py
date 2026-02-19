#!/usr/bin/env python3

from celery import Celery
import time
import os
from process_data import process_file

app = Celery('tasks')
app.config_from_object('celeryconfig')

@app.task
def process_data_file(filename):
    """
    Process a data file with the specified filename.
    
    Args:
        filename: Name of the file to process
        
    Returns:
        Success message upon completion
    """
    filepath = os.path.join('/opt/celery_app/data', filename)
    process_file(filepath)
    return f"Successfully processed {filename}"