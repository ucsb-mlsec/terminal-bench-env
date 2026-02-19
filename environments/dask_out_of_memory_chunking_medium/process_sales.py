#!/usr/bin/env python3

import pandas as pd
import json
import sys

# Current implementation - loads entire dataset into memory
# This approach fails with large datasets that exceed available RAM

def process_sales_data():
    """
    Process sales data and calculate aggregated metrics.
    WARNING: This implementation loads all data into memory at once.
    """
    
    print("Loading entire dataset into memory...")
    
    # Load the ENTIRE CSV file into memory at once - problematic for large files
    df = pd.read_csv('/workspace/data/sales_q1.csv')
    
    print(f"Loaded {len(df)} records into memory")
    
    # Create multiple full copies of the DataFrame - inefficient memory usage
    df_daily = df.copy()  # Full copy for daily aggregation
    df_category = df.copy()  # Full copy for category aggregation
    df_region = df.copy()  # Full copy for region aggregation
    
    print("Processing daily totals...")
    # Calculate daily sales totals - operates on full dataset in memory
    daily_totals = df_daily.groupby('date')['amount'].sum().to_dict()
    
    print("Processing category totals...")
    # Calculate category totals - operates on full dataset in memory
    category_totals = df_category.groupby('product_category')['amount'].sum().to_dict()
    
    print("Processing region totals...")
    # Calculate region totals - operates on full dataset in memory
    region_totals = df_region.groupby('region')['amount'].sum().to_dict()
    
    # Round all values to 2 decimal places
    daily_totals = {k: round(v, 2) for k, v in daily_totals.items()}
    category_totals = {k: round(v, 2) for k, v in category_totals.items()}
    region_totals = {k: round(v, 2) for k, v in region_totals.items()}
    
    # Prepare results dictionary
    results = {
        "daily_totals": daily_totals,
        "category_totals": category_totals,
        "region_totals": region_totals
    }
    
    # Write results to JSON file
    print("Writing results to file...")
    with open('/workspace/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Processing complete!")
    print(f"Daily totals: {len(daily_totals)} entries")
    print(f"Category totals: {len(category_totals)} entries")
    print(f"Region totals: {len(region_totals)} entries")

if __name__ == "__main__":
    try:
        process_sales_data()
    except MemoryError:
        print("ERROR: Out of memory! Dataset is too large to fit in RAM.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)