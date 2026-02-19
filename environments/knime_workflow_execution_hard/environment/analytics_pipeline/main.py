#!/usr/bin/env python3

import os
import json
from data_load import load_transactions
from data_processor import calculate_customer_totals, filter_high_value_customers
from output_generator import generate_output


def main():
    """
    Main orchestration function for the analytics pipeline.
    Processes customer transaction data to identify high-value customers.
    """
    print("Starting analytics pipeline...")
    
    # Step 1: Load transaction data
    print("Loading transaction data...")
    transactions = load_transactions('/data/transactions.csv')
    
    if transactions is None or len(transactions) == 0:
        print("Error: No transactions loaded")
        return
    
    print(f"Loaded {len(transactions)} transaction records")
    
    # Step 2: Calculate customer totals
    print("Calculating customer totals...")
    customer_totals = calculate_customer_totals(transactions)
    
    # Step 3: Filter high-value customers (threshold: 5000.0)
    print("Filtering high-value customers...")
    high_value_threshold = 5000.0
    high_value_customers_data = filter_high_value_customers(customer_totals, high_value_threshold)
    
    # Step 4: Count results
    total_processed = len(transactions)
    high_value_customers = len(high_value_customers_data)
    
    print(f"Total processed: {total_processed}")
    print(f"High-value customers: {high_value_customers}")
    
    # Step 5: Generate output
    print("Generating output...")
    output_path = '/solution/results.json'
    generate_output(high_value_customers, total_processed, output_path)
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()