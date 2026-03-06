#!/usr/bin/env python3

import pandas as pd
import os
import sys

# Task-based pipeline system mimicking Airflow's execution pattern
# Uses XCom-like in-memory dictionary for passing data between tasks

# In-memory XCom storage (causes memory issues with large datasets)
xcom = {}

def extract_task():
    """
    Extract task: Read CSV file and store entire dataset in memory
    """
    print("Starting extract task...")
    input_file = '/home/user/pipeline/input/transactions.csv'
    
    # Read the entire CSV file into memory
    df = pd.read_csv(input_file)
    
    # Store entire dataset in XCom dictionary (memory-intensive approach)
    xcom['extracted_data'] = df.copy()
    
    print(f"Extracted {len(df)} records")
    print(f"Memory stored in xcom: {sys.getsizeof(xcom['extracted_data'])} bytes")
    return True

def transform_task():
    """
    Transform task: Retrieve data from XCom, transform it, and store back in XCom
    """
    print("Starting transform task...")
    
    # Retrieve entire dataset from XCom (requires unpickling large object)
    df = xcom['extracted_data'].copy()
    
    # Perform transformations
    # Convert amount column to float if it exists
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Filter records where status is 'Completed' if column exists
    if 'status' in df.columns:
        df = df[df['status'] == 'Completed']
    
    # Add calculated column (e.g., processing fee)
    if 'amount' in df.columns:
        df['processing_fee'] = df['amount'] * 0.02
    
    # Store entire transformed dataset back in XCom (another memory-intensive copy)
    xcom['transformed_data'] = df.copy()
    
    print(f"Transformed {len(df)} records")
    print(f"Memory stored in xcom: {sys.getsizeof(xcom['transformed_data'])} bytes")
    return True

def load_task():
    """
    Load task: Retrieve transformed data from XCom and write to output file
    """
    print("Starting load task...")
    
    # Retrieve entire dataset from XCom (another unpickling operation)
    df = xcom['transformed_data']
    
    # Ensure output directory exists
    output_dir = '/home/user/pipeline/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to CSV
    output_file = '/home/user/pipeline/output/processed_transactions.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Loaded {len(df)} records to {output_file}")
    return True

def main():
    """
    Main execution function - runs tasks in sequence
    """
    print("="*60)
    print("Starting Data Pipeline Execution")
    print("="*60)
    
    try:
        # Execute tasks in sequence: extract → transform → load
        extract_task()
        print("-"*60)
        
        transform_task()
        print("-"*60)
        
        load_task()
        print("-"*60)
        
        print("Pipeline execution completed successfully!")
        
    except MemoryError as e:
        print(f"MEMORY ERROR: Pipeline failed due to memory issues - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Pipeline failed - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()