#!/usr/bin/env python3

import json
import os
import sys

def load_config():
    """Load feature store configuration"""
    config_path = '/opt/feature_store/feature_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_customer_batch():
    """Load customer IDs from batch file"""
    batch_path = '/opt/feature_store/customer_batch.txt'
    with open(batch_path, 'r') as f:
        customers = [line.strip() for line in f if line.strip()]
    return customers

def load_features(date):
    """Load feature data for a specific date"""
    config = load_config()
    file_pattern = config.get('feature_file_pattern', 'features_{date}.json')
    filename = file_pattern.replace('{date}', date)
    
    # Bug 1: Incorrect path construction - using string concatenation with wrong separator
    feature_path = '/opt/feature_store/' + filename
    
    with open(feature_path, 'r') as f:
        feature_data = json.load(f)
    return feature_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve_features.py <reference_date>")
        sys.exit(1)
    
    reference_date = sys.argv[1]
    
    # Load configuration and customer batch
    config = load_config()
    customers = load_customer_batch()
    
    # Load feature data for the reference date
    all_features = load_features(reference_date)
    
    # Build output structure
    output = {
        'reference_date': reference_date,
        'features': {}
    }
    
    # Extract features for each customer in the batch
    for customer_id in customers:
        # Bug 2: Wrong variable name - using 'all_feature' instead of 'all_features'
        if customer_id in all_feature:
            customer_features = all_features[customer_id]
            
            # Bug 3: Not extracting the actual feature values, storing entire object
            output['features'][customer_id] = {
                'revenue': customer_features['revenue'],
                'visits': customer_features['visits']
            }
    
    # Save output to file
    output_path = '/opt/feature_store/output.json'
    f = open(output_path, 'w')
    json.dump(output, f, indent=2)
    # Bug 4: Missing file close - file handle not closed
    
    print(f"Features retrieved successfully for {len(output['features'])} customers")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    main()