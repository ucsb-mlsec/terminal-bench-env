#!/usr/bin/env python3

import pandas as pd
import pandera as pa
import numpy as np
from schemas.stage3_schema import Stage3Schema

def process_stage3():
    """
    Process stage 3: Aggregate transaction data by customer and product category.
    Contains intentional bugs that cause schema validation failures.
    """
    # Read stage 2 output
    df = pd.read_csv('/workspace/data/stage2_output.csv')
    
    # BUG 1: Missing 'product_category' from group by - loses important grouping dimension
    # BUG 2: Using 'mean' instead of 'sum' for total_amount
    # BUG 3: Using 'sum' instead of 'count' for transaction_count
    # BUG 4: Missing 'avg_quantity' column entirely
    # BUG 5: Wrong column name 'maximum_amount' instead of 'max_amount'
    # BUG 6: Missing 'min_amount' column
    # BUG 7: Aggregated values stored as strings instead of numeric types
    aggregated = df.groupby(['customer_id']).agg({
        'amount': ['mean', 'max', 'min'],  # Wrong function for total_amount
        'quantity': 'sum'  # Wrong - should be transaction count
    }).reset_index()
    
    # BUG 8: Incorrect column flattening and naming
    aggregated.columns = ['customer_id', 'total_amount', 'maximum_amount', 'min_amt', 'transaction_count']
    
    # BUG 9: Converting numeric columns to strings (type mismatch)
    aggregated['total_amount'] = aggregated['total_amount'].astype(str)
    aggregated['transaction_count'] = aggregated['transaction_count'].astype(str)
    
    # BUG 10: Missing avg_quantity calculation entirely (required column)
    
    # BUG 11: Not including product_category in final output (required column)
    
    # BUG 12: Wrong column name 'maximum_amount' should be 'max_amount'
    # Also 'min_amt' should be 'min_amount'
    
    # Validate against stage 3 schema
    Stage3Schema.validate(aggregated)
    
    # Save output
    aggregated.to_csv('/workspace/data/stage3_output.csv', index=False)
    
    return aggregated

if __name__ == '__main__':
    print("Processing Stage 3: Aggregation...")
    result = process_stage3()
    print(f"Stage 3 complete. Processed {len(result)} aggregated records.")
    print("Output saved to /workspace/data/stage3_output.csv")