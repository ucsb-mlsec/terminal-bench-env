#!/usr/bin/env python3

import os
import sys
import csv
import json
import time
from datetime import datetime
from collections import defaultdict

# Configuration
DATA_DIR = '/opt/app/data'
OUTPUT_DIR = '/opt/app/output'
BATCH_SIZE = 1000

def load_data_files(directory):
    """
    Load all CSV data files from the specified directory.
    Returns a list of file paths.
    """
    data_files = []
    if not os.path.exists(directory):
        print(f"Warning: Data directory {directory} does not exist")
        return data_files
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            data_files.append(filepath)
    
    print(f"Found {len(data_files)} data files to process")
    return data_files


def read_csv_records(filepath):
    """
    Read records from a CSV file and return as list of dictionaries.
    """
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return records


def validate_entries(records):
    """
    Validate data entries for completeness and correctness.
    Returns tuple of (valid_records, invalid_count).
    """
    valid_records = []
    invalid_count = 0
    
    required_fields = ['id', 'timestamp', 'value', 'category']
    
    for record in records:
        is_valid = True
        
        # Check required fields
        for field in required_fields:
            if field not in record or not record[field]:
                is_valid = False
                break
        
        # Validate value is numeric
        if is_valid:
            try:
                float(record['value'])
            except ValueError:
                is_valid = False
        
        if is_valid:
            valid_records.append(record)
        else:
            invalid_count += 1
    
    return valid_records, invalid_count


def transform_records(records):
    """
    Transform and enrich records with additional computed fields.
    """
    transformed = []
    
    for record in records:
        new_record = record.copy()
        
        # Add computed fields
        new_record['value_squared'] = float(record['value']) ** 2
        new_record['category_upper'] = record['category'].upper()
        new_record['processed_at'] = datetime.now().isoformat()
        
        transformed.append(new_record)
    
    return transformed


def calculate_metrics(records):
    """
    Calculate various metrics and statistics from the records.
    """
    metrics = {
        'total_records': len(records),
        'sum_values': 0.0,
        'avg_value': 0.0,
        'category_counts': defaultdict(int)
    }
    
    for record in records:
        value = float(record['value'])
        metrics['sum_values'] += value
        metrics['category_counts'][record['category']] += 1
    
    if metrics['total_records'] > 0:
        metrics['avg_value'] = metrics['sum_values'] / metrics['total_records']
    
    return metrics


def process_large_dataset(data_files):
    """
    Main processing function that handles large datasets.
    This function performs multiple passes over the data with nested operations.
    """
    all_records = []
    
    # Load all data files
    for filepath in data_files:
        records = read_csv_records(filepath)
        all_records.extend(records)
    
    print(f"Loaded {len(all_records)} total records")
    
    # Validate all records
    valid_records, invalid_count = validate_entries(all_records)
    print(f"Valid records: {len(valid_records)}, Invalid: {invalid_count}")
    
    # Transform records
    transformed_records = transform_records(valid_records)
    
    # Perform expensive nested operations (bottleneck simulation)
    processed_data = []
    for i, record in enumerate(transformed_records):
        enriched_record = record.copy()
        
        # Nested loop - expensive operation
        matching_count = 0
        total_matching_value = 0.0
        
        for other_record in transformed_records:
            if record['category'] == other_record['category']:
                matching_count += 1
                total_matching_value += float(other_record['value'])
        
        # Add aggregated fields
        enriched_record['category_match_count'] = matching_count
        enriched_record['category_total_value'] = total_matching_value
        
        if matching_count > 0:
            enriched_record['category_avg_value'] = total_matching_value / matching_count
        else:
            enriched_record['category_avg_value'] = 0.0
        
        processed_data.append(enriched_record)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(transformed_records)} records")
    
    # Calculate final metrics
    metrics = calculate_metrics(processed_data)
    
    return processed_data, metrics


def write_output(data, metrics, output_dir):
    """
    Write processed data and metrics to output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Write processed data
    output_file = os.path.join(output_dir, 'processed_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    # Write metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Output written to {output_dir}")


def main():
    """
    Main execution function.
    """
    print("Starting data processing worker...")
    start_time = time.time()
    
    # Load data files
    data_files = load_data_files(DATA_DIR)
    
    if not data_files:
        print("No data files found. Exiting.")
        return 1
    
    # Process the dataset
    processed_data, metrics = process_large_dataset(data_files)
    
    # Write output
    write_output(processed_data, metrics, OUTPUT_DIR)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Total records processed: {len(processed_data)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())