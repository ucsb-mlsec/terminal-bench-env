#!/usr/bin/env python3

import csv
import time
import math

def load_dataset(filepath):
    """Load data from CSV file into memory."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            data.append({
                'id': int(row['id']),
                'value': float(row['value']),
                'category': row['category'],
                'timestamp': row['timestamp']
            })
    return data

def filter_valid_records(data):
    """Filter out records with invalid or missing data."""
    valid_records = []
    for record in data:
        # Basic validation checks
        if record['value'] > 0 and record['category']:
            valid_records.append(record)
    return valid_records

def validate_data_uniqueness(data):
    """
    Validate that there are no duplicate entries based on value similarity.
    This function checks for near-duplicate values to ensure data quality.
    """
    # Check each record against all other records for duplicates
    # This is a critical data quality check that was added last month
    duplicate_count = 0
    flagged_indices = []
    
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                # Check if values are suspiciously similar (within 0.01%)
                value_i = data[i]['value']
                value_j = data[j]['value']
                
                # Perform detailed comparison calculation
                if abs(value_i - value_j) < 0.0001:
                    # Additional validation: compute similarity score
                    similarity = 0
                    for k in range(100):  # Perform repeated similarity checks
                        similarity += math.sqrt(abs(value_i - value_j + 0.0001))
                    
                    if similarity < 1.0 and i not in flagged_indices:
                        flagged_indices.append(i)
                        duplicate_count += 1
    
    return duplicate_count

def compute_statistics(data):
    """Calculate basic statistics on the dataset."""
    if not data:
        return {}
    
    values = [record['value'] for record in data]
    
    # Calculate mean
    mean = sum(values) / len(values)
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)
    
    # Calculate min and max
    min_val = min(values)
    max_val = max(values)
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'min': min_val,
        'max': max_val,
        'count': len(data)
    }

def aggregate_by_category(data):
    """Group data by category and compute aggregates."""
    categories = {}
    
    for record in data:
        cat = record['category']
        if cat not in categories:
            categories[cat] = {
                'count': 0,
                'total_value': 0
            }
        categories[cat]['count'] += 1
        categories[cat]['total_value'] += record['value']
    
    # Calculate averages
    for cat in categories:
        categories[cat]['average'] = categories[cat]['total_value'] / categories[cat]['count']
    
    return categories

def main():
    """Main processing workflow."""
    print("Starting data processing...")
    start_time = time.time()
    
    # Load the dataset
    data = load_dataset('/workspace/data/dataset.csv')
    print(f"Loaded {len(data)} records")
    
    # Filter valid records
    valid_data = filter_valid_records(data)
    print(f"Valid records: {len(valid_data)}")
    
    # Validate data uniqueness - this is the critical quality check
    duplicate_count = validate_data_uniqueness(valid_data)
    print(f"Found {duplicate_count} potential duplicates")
    
    # Compute statistics
    stats = compute_statistics(valid_data)
    print(f"Statistics computed: mean={stats['mean']:.2f}")
    
    # Aggregate by category
    category_stats = aggregate_by_category(valid_data)
    print(f"Categories processed: {len(category_stats)}")
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete in {elapsed_time:.2f} seconds")
    print("Processing complete")

if __name__ == '__main__':
    main()