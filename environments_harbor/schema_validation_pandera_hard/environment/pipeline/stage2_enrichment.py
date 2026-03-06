#!/usr/bin/env python3

import pandas as pd
import pandera as pa
from schemas.stage2_schema import Stage2Schema

def process_stage2():
    """
    Enriches stage1 data with customer information.
    Contains intentional bugs for schema validation failures.
    """
    # Read stage1 output
    df = pd.read_csv('/workspace/data/stage1_output.csv')
    
    # Bug 1: Generate customer names with incorrect format (missing prefix)
    df['customer_name'] = df['customer_id'].apply(lambda x: f"{x}_Smith")
    
    # Bug 2: Incorrect segment classification logic (using wrong thresholds)
    def classify_segment(amount):
        if amount > 500:
            return 'GOLD'  # Bug: Wrong case
        elif amount > 200:
            return 'Silver'
        else:
            return 'bronze'  # Bug: Wrong case
    
    df['customer_segment'] = df['amount'].apply(classify_segment)
    
    # Bug 3: Region assignment with invalid values
    regions = ['North', 'South', 'East', 'West', 'Central']  # Bug: Central not valid
    df['region'] = df.index % 5
    df['region'] = df['region'].apply(lambda x: regions[x])
    
    # Bug 4: Discount rate calculation with values outside valid range
    # Should be 0-1, but this produces values outside that range
    df['discount_rate'] = (df['amount'] / 100).round(2)  # Bug: Can exceed 1.0
    
    # Bug 5: Missing 'loyalty_points' column that might be required
    
    # Bug 6: Wrong data type for customer_id (should remain int but converting to string)
    df['customer_id'] = df['customer_id'].astype(str)
    
    # Bug 7: Creating extra column not in schema
    df['extra_field'] = 'unnecessary'
    
    # Bug 8: Incorrect transaction_date format
    df['transaction_date'] = pd.to_datetime(df['transaction_date']).dt.strftime('%Y/%m/%d')  # Bug: Wrong format
    
    # Validate against schema
    Stage2Schema.validate(df)
    
    # Save output
    df.to_csv('/workspace/data/stage2_output.csv', index=False)
    print("Stage 2 processing completed successfully")
    return df

if __name__ == '__main__':
    process_stage2()