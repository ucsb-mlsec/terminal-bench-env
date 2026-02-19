#!/usr/bin/env python3

import pandas as pd
import yaml
import sys

def validate_data():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Incorrect parameter path - should be params['validation']
    validation_params = params['validate']
    
    # Read input data
    df = pd.read_csv('data/sales_transactions.csv')
    
    # Check for missing values
    missing_threshold = validation_params['missing_threshold']
    initial_rows = len(df)
    df = df.dropna(thresh=len(df.columns) * (1 - missing_threshold))
    
    # Validate quantity - wrong column name 'qty' instead of 'quantity'
    min_quantity = validation_params['min_quantity']
    df = df[df['qty'] >= min_quantity]
    
    # Validate date format
    date_format = validation_params['date_format']
    try:
        df['date'] = pd.to_datetime(df['date'], format=date_format)
    except:
        # Missing error handling - just pass silently
        pass
    
    # Save validated data
    df.to_csv('data/validated_data.csv', index=False)
    
    print(f"Validation complete. Rows processed: {initial_rows}, Rows after validation: {len(df)}")

if __name__ == '__main__':
    validate_data()