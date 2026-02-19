#!/usr/bin/env python3

import csv
import sys
from multiprocessing import Pool

# Data processor class that will cause pickle errors
# Instance methods cannot be serialized across process boundaries
class DataProcessor:
    def __init__(self, multiplier, threshold):
        self.multiplier = multiplier
        self.threshold = threshold
        self.processed_count = 0
    
    def process_row(self, row):
        # Process a single row of data
        # Apply multiplier to value and check against threshold
        value = float(row['value'])
        processed_value = value * self.multiplier
        
        # Apply threshold logic
        if processed_value > self.threshold:
            result = processed_value * 1.1
        else:
            result = processed_value * 0.9
        
        return {
            'id': row['id'],
            'original_value': value,
            'processed_value': result,
            'category': row['category']
        }
    
    def calculate_statistics(self, results):
        # Calculate basic statistics from processed results
        total = sum(r['processed_value'] for r in results)
        count = len(results)
        average = total / count if count > 0 else 0
        return total, count, average

if __name__ == '__main__':
    # Initialize the processor with configuration
    processor = DataProcessor(multiplier=2.5, threshold=100.0)
    
    # Read all data from CSV file
    print("Reading data from /workspace/input.csv...")
    data_rows = []
    with open('/workspace/input.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_rows.append(row)
    
    print(f"Loaded {len(data_rows)} rows from CSV")
    
    # Process data using multiprocessing
    # This will FAIL because instance methods cannot be pickled
    print("Processing data with multiprocessing...")
    with Pool(processes=4) as pool:
        results = pool.map(processor.process_row, data_rows)
    
    # Calculate statistics from results
    total, count, average = processor.calculate_statistics(results)
    
    print(f"Processing complete!")
    print(f"Rows processed: {count}")
    print(f"Total sum: {total:.2f}")
    print(f"Average: {average:.2f}")