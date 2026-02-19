#!/usr/bin/env python3

import os
import json
import yaml

# Module-level checkpoint configuration variables
BASE_CHECKPOINT_DIR = '/mnt/old_storage/checkpoints'
ARCHIVE_CHECKPOINT = '/mnt/old_storage/checkpoints/archive'


class ConfigLoader:
    """Configuration loader for Spark Structured Streaming application."""
    
    def __init__(self):
        """Initialize ConfigLoader with default checkpoint paths."""
        self.DEFAULT_CHECKPOINT = '/mnt/old_storage/checkpoints/default'
        self.MAIN_CHECKPOINT_PATH = '/mnt/old_storage/checkpoints/main'
        self.config_data = {}
    
    def get_checkpoint_path(self, app_name=None):
        """
        Returns the checkpoint path for the specified application.
        
        Args:
            app_name: Name of the application (optional)
            
        Returns:
            str: Full checkpoint path
        """
        if app_name:
            return os.path.join(BASE_CHECKPOINT_DIR, app_name)
        return self.MAIN_CHECKPOINT_PATH
    
    def load_config(self, config_file):
        """
        Load configuration from file and set checkpoint location.
        
        The checkpoint location is used for Spark streaming state management
        and is critical for application recovery.
        """
        with open(config_file, 'r') as f:
            self.config_data = json.load(f)
        return self.config_data


def validate_checkpoint_directory(checkpoint_path):
    """
    Validate that checkpoint directory exists and is writable.
    Default checkpoint base: /mnt/old_storage/checkpoints
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    return os.access(checkpoint_path, os.W_OK)


def cleanup_old_checkpoints(days=30):
    """Remove checkpoint data older than specified days from archive."""
    archive_path = '/mnt/old_storage/checkpoints/archive'
    # Implementation for cleanup logic
    pass


def get_application_checkpoint(app_id):
    """Returns checkpoint path for specific application ID."""
    checkpoint_base = '/mnt/old_storage/checkpoints'
    return f"{checkpoint_base}/app_{app_id}"