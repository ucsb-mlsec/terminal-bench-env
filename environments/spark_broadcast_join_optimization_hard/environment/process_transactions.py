#!/usr/bin/env python3

import pandas as pd
import os

def main():
    # Set the data directory path
    data_dir = '/tmp/workspace/data'
    
    # Load transaction data
    print("Loading transaction data...")
    transactions = pd.read_csv(os.path.join(data_dir, 'transactions.csv'))
    print(f"Loaded {len(transactions)} transaction records")
    
    # Load reference tables
    print("\nLoading reference data...")
    customers = pd.read_csv(os.path.join(data_dir, 'customers.csv'))
    print(f"Loaded {len(customers)} customer records")
    
    products = pd.read_csv(os.path.join(data_dir, 'products.csv'))
    print(f"Loaded {len(products)} product records")
    
    stores = pd.read_csv(os.path.join(data_dir, 'stores.csv'))
    print(f"Loaded {len(stores)} store records")
    
    # Perform joins without any optimization
    print("\nPerforming joins...")
    
    # Join with customers
    print("Joining with customers...")
    transactions_with_customers = transactions.merge(customers, on='customer_id', how='left')
    
    # Join with products
    print("Joining with products...")
    transactions_with_products = transactions_with_customers.merge(products, on='product_id', how='left')
    
    # Join with stores
    print("Joining with stores...")
    final_data = transactions_with_products.merge(stores, on='store_id', how='left')
    
    # Basic processing
    print("\nProcessing complete!")
    print(f"Final dataset shape: {final_data.shape}")
    print(f"Total columns: {len(final_data.columns)}")
    
    # Display sample statistics
    if 'amount' in final_data.columns:
        print(f"Total transaction amount: ${final_data['amount'].sum():,.2f}")
        print(f"Average transaction amount: ${final_data['amount'].mean():,.2f}")
    
    print("\nFirst few records of enriched data:")
    print(final_data.head())

if __name__ == "__main__":
    main()