#!/usr/bin/env python3

import pandas as pd
import pandera as pa
from schemas.stage1_schema import stage1_schema

def process_stage1():
    """
    Process stage 1: Ingestion of raw transaction data.
    This version has intentional bugs causing schema violations.
    """
    # Read raw transactions
    df = pd.read_csv('/workspace/data/raw_transactions.csv')
    
    # BUG 1: Not converting transaction_id to string (schema expects string)
    # Should be: df['transaction_id'] = df['transaction_id'].astype(str)
    
    # BUG 2: Not converting amount to float (keeping as string/object)
    # Should be: df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # BUG 3: Not parsing transaction_date as datetime
    # Should be: df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    
    # BUG 4: Not renaming 'cust_id' to 'customer_id' as schema expects
    # Should be: df = df.rename(columns={'cust_id': 'customer_id'})
    
    # BUG 5: Not handling missing values in required fields
    # Should be: df = df.dropna(subset=['transaction_id', 'customer_id', 'amount'])
    
    # BUG 6: Not filtering out negative amounts (schema likely has constraints)
    # Should be: df = df[df['amount'] > 0]
    
    # BUG 7: Not converting customer_id to string type
    # Should be: df['customer_id'] = df['customer_id'].astype(str)
    
    # BUG 8: Not stripping whitespace from string columns
    # Should be: df['transaction_type'] = df['transaction_type'].str.strip()
    
    # BUG 9: Not handling case sensitivity in transaction_type
    # Should be: df['transaction_type'] = df['transaction_type'].str.lower()
    
    # Validate against schema (this will fail due to bugs above)
    validated_df = stage1_schema.validate(df)
    
    # Save output
    validated_df.to_csv('/workspace/data/stage1_output.csv', index=False)
    print("Stage 1 completed successfully")
    
    return validated_df

if __name__ == '__main__':
    process_stage1()