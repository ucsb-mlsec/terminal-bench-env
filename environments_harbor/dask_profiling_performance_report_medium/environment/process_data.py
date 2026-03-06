#!/usr/bin/env python3

import dask.dataframe as dd
import glob
import pandas as pd

def main():
    # Read all CSV files from the data directory
    csv_files = glob.glob('/workspace/data/*.csv')
    
    if not csv_files:
        print("No CSV files found in /workspace/data/")
        return
    
    # Load CSV files into Dask dataframe
    ddf = dd.read_csv('/workspace/data/*.csv', 
                      dtype={'customer_id': 'int64', 
                             'transaction_date': 'object',
                             'amount': 'float64',
                             'category': 'object'})
    
    # Perform data transformations and aggregations
    
    # 1. Calculate total transactions per customer
    customer_totals = ddf.groupby('customer_id')['amount'].sum().compute()
    print("Top 5 customers by transaction amount:")
    print(customer_totals.nlargest(5))
    print()
    
    # 2. Calculate average transaction amount by category
    category_avg = ddf.groupby('category')['amount'].mean().compute()
    print("Average transaction amount by category:")
    print(category_avg)
    print()
    
    # 3. Filter high-value transactions (amount > 100)
    high_value = ddf[ddf['amount'] > 100].compute()
    print(f"Number of high-value transactions (>100): {len(high_value)}")
    print()
    
    # 4. Calculate total revenue by category
    category_totals = ddf.groupby('category')['amount'].sum().compute()
    print("Total revenue by category:")
    print(category_totals.sort_values(ascending=False))
    print()
    
    # 5. Count transactions per customer
    transaction_counts = ddf.groupby('customer_id').size().compute()
    print(f"Average transactions per customer: {transaction_counts.mean():.2f}")
    print()
    
    # 6. Overall statistics
    total_amount = ddf['amount'].sum().compute()
    total_transactions = len(ddf)
    print(f"Total amount processed: ${total_amount:,.2f}")
    print(f"Total number of transactions: {total_transactions}")

if __name__ == "__main__":
    main()