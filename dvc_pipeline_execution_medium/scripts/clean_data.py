#!/usr/bin/env python3

import pandas as pd
import os
import sys

def clean_data():
    """
    Clean raw transaction data by removing invalid transactions.
    """
    try:
        # Read raw transaction data
        input_file = 'data/raw_transactions.csv'
        output_file = 'data/cleaned_transactions.csv'
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        
        # Read the raw data
        df = pd.read_csv(input_file)
        
        # Get initial count
        initial_count = len(df)
        
        # Remove rows where amount is missing (NaN/empty)
        df = df.dropna(subset=['amount'])
        
        # Remove rows where amount is negative or zero
        df = df[df['amount'] > 0]
        
        # Remove rows where status is not 'completed'
        df = df[df['status'] == 'completed']
        
        # Remove any rows with any remaining missing values
        df = df.dropna()
        
        # Get final count
        valid_count = len(df)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f'Data cleaning complete. {valid_count} valid transactions saved.')
        
        return valid_count
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    clean_data()