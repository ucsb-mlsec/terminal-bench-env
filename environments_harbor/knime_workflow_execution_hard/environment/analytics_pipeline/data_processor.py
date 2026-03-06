#!/usr/bin/env python3

import pandas as pd
import sys

def calculate_customer_totals(df):
    """
    Groups transactions by customer_id and calculates total spending.
    Contains bug: uses 'Sum' instead of 'sum' causing AttributeError
    Contains bug: references 'ammount' instead of 'amount'
    """
    try:
        # Bug: wrong column name 'ammount' instead of 'amount'
        customer_totals = df.groupby('customer_id')['ammount'].Sum()
        return customer_totals.reset_index()
    except Exception as e:
        print(f"Error calculating customer totals: {e}")
        raise

def filter_high_value_customers(df, threshold):
    """
    Filters customers whose total spending is above the threshold.
    Contains bug: compares string to float without conversion
    """
    high_value = []
    
    for idx, row in df.iterrows():
        # Bug: row['ammount'] might be string, comparing to float without conversion
        if row['ammount'] > threshold:
            high_value.append(row['customer_id'])
    
    return high_value

def load_transaction_data(filepath):
    """
    Loads transaction data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def process_pipeline(input_file, threshold=5000.0):
    """
    Main pipeline processing function.
    """
    # Load data
    df = load_transaction_data(input_file)
    total_processed = len(df)
    
    # Calculate customer totals
    customer_totals = calculate_customer_totals(df)
    
    # Filter high value customers
    high_value_customers = filter_high_value_customers(customer_totals, threshold)
    
    return {
        'high_value_customers': len(high_value_customers),
        'total_processed': total_processed
    }

if __name__ == "__main__":
    result = process_pipeline('/data/transactions.csv')
    print(f"Processed {result['total_processed']} transactions")
    print(f"Found {result['high_value_customers']} high-value customers")