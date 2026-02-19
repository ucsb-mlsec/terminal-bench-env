#!/usr/bin/env python3

import json
import csv
import io
import tempfile

# Memory leak: global list that accumulates temp file paths indefinitely
temp_files = []

def process_task_data(task_id, data):
    """
    Process task data and write to temporary file.
    Memory leak: file handle opened but never closed
    """
    # Memory leak: file opened without closing
    file_handle = open(f'/tmp/task_{task_id}.tmp', 'w')
    file_handle.write(json.dumps(data))
    file_handle.flush()
    
    # Memory leak: accumulating paths without cleanup
    temp_files.append(f'/tmp/task_{task_id}.tmp')
    
    return f'/tmp/task_{task_id}.tmp'

def load_config(config_path):
    """
    Load configuration from JSON file.
    Memory leak: file opened without context manager or explicit close
    """
    # Memory leak: unclosed file handle
    config_file = open(config_path, 'r')
    config_data = config_file.read()
    parsed_config = json.loads(config_data)
    
    return parsed_config

class DataBuffer:
    """
    Buffer for storing data.
    Memory leak: unbounded growth without cleanup
    """
    def __init__(self):
        self.buffer = []
    
    def add_data(self, data):
        """
        Add data to buffer.
        Memory leak: no size limits or eviction policy
        """
        self.buffer.append(data)
    
    def get_all(self):
        """
        Get all buffered data.
        Memory leak: returns buffer but doesn't clear it
        """
        return self.buffer

# Memory leak: global buffer instance that accumulates data
global_buffer = DataBuffer()

def record_task_metrics(task_id, metrics):
    """
    Record metrics for a task in the global buffer.
    """
    metric_data = {
        'task_id': task_id,
        'metrics': metrics,
        'timestamp': None
    }
    global_buffer.add_data(metric_data)