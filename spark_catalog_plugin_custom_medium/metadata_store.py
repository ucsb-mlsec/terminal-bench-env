#!/usr/bin/env python3
"""
Metadata storage layer for table definitions.
Handles reading and writing table metadata files.
"""

import os
import os.path


class MetadataStore:
    """
    Storage backend for table metadata files.
    Manages .meta files containing table definitions.
    """
    
    def __init__(self, metadata_dir):
        """
        Initialize the metadata store.
        
        Args:
            metadata_dir (str): Path to the directory containing metadata files
        """
        self.metadata_dir = metadata_dir
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
    
    def list_tables(self):
        """
        Get a list of all registered tables.
        
        Returns:
            list: List of table names (without .meta extension)
        """
        tables = []
        
        if not os.path.exists(self.metadata_dir):
            return tables
        
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith('.meta'):
                # Remove .meta extension to get table name
                table_name = filename[:-5]
                tables.append(table_name)
        
        return sorted(tables)
    
    def read_metadata(self, table_name):
        """
        Read metadata for a specific table.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            dict: Dictionary of metadata key-value pairs, or None if not found
        """
        filepath = os.path.join(self.metadata_dir, f"{table_name}.meta")
        
        if not os.path.exists(filepath):
            return None
        
        metadata = {}
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error reading metadata for {table_name}: {e}")
            return None
        
        return metadata
    
    def write_metadata(self, table_name, metadata):
        """
        Write metadata for a table.
        
        Args:
            table_name (str): Name of the table
            metadata (dict): Dictionary of metadata key-value pairs
        """
        filepath = os.path.join(self.metadata_dir, f"{table_name}.meta")
        
        try:
            with open(filepath, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}={value}\n")
        except Exception as e:
            print(f"Error writing metadata for {table_name}: {e}")
            raise
    
    def delete_metadata(self, table_name):
        """
        Delete metadata file for a table.
        
        Args:
            table_name (str): Name of the table
        """
        filepath = os.path.join(self.metadata_dir, f"{table_name}.meta")
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting metadata for {table_name}: {e}")