#!/usr/bin/env python3

"""
Current INEFFICIENT implementation of employee query script.
This script reads ALL data into memory first, then filters in Python.
This causes excessive memory usage and slow performance.
"""

import pandas as pd

def query_employees():
    # PROBLEM: Reading entire Parquet file without any filters
    # This loads all 100,000 rows into memory
    print("Reading entire Parquet file into memory...")
    df = pd.read_parquet('/data/employees.parquet')
    
    print(f"Loaded {len(df)} rows into memory")
    
    # PROBLEM: Filtering happens AFTER loading all data
    # This is inefficient - we've already read everything from disk
    print("Applying filters in Python...")
    filtered_df = df[(df['department'] == 'Engineering') & (df['salary'] > 75000)]
    
    print(f"Found {len(filtered_df)} matching employees")
    print("\nSample results:")
    print(filtered_df.head())
    
    return filtered_df

if __name__ == "__main__":
    # Execute the inefficient query
    result = query_employees()
    print(f"\nTotal matching records: {len(result)}")