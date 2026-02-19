#!/usr/bin/env python3
import csv
from typing import List, Dict, Optional, Any

class CSVReader:
    """
    Custom CSV data reader for querying CSV files.
    
    Current implementation: INEFFICIENT - loads all data before filtering
    
    Example usage:
        reader = CSVReader('/path/to/data.csv')
        results = reader.query(filters={'id': 5000})
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the CSV reader with a file path.
        
        Args:
            file_path: Path to the CSV file to read
        """
        self.file_path = file_path
    
    def query(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the CSV file and return matching rows.
        
        PROBLEM: Currently loads ALL rows into memory first, then filters.
        This is inefficient for large files with selective filters.
        
        Args:
            filters: Dictionary of column:value pairs to filter by (currently ignored during read)
        
        Returns:
            List of dictionaries representing matching rows
        """
        all_rows = []
        
        try:
            # INEFFICIENT: Read ALL rows from CSV file into memory first
            with open(self.file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Load every single row - no filtering during read!
                for row in reader:
                    # Convert numeric columns for proper comparison
                    if 'id' in row:
                        row['id'] = int(row['id'])
                    if 'timestamp' in row:
                        row['timestamp'] = int(row['timestamp'])
                    if 'temperature' in row:
                        row['temperature'] = float(row['temperature'])
                    if 'pressure' in row:
                        row['pressure'] = float(row['pressure'])
                    
                    all_rows.append(row)
            
            # OPTIMIZATION NEEDED: Filters are applied AFTER loading all data
            # This wastes memory and I/O when filters are selective
            if filters:
                filtered_rows = []
                for row in all_rows:
                    match = True
                    for column, value in filters.items():
                        if column not in row or row[column] != value:
                            match = False
                            break
                    if match:
                        filtered_rows.append(row)
                return filtered_rows
            
            return all_rows
            
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return []
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []
    
    # Alias for backwards compatibility
    def read(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Alias for query method"""
        return self.query(filters)