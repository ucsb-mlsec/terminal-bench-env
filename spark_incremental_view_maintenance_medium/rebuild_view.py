#!/usr/bin/env python3
"""
rebuild_view.py - Current inefficient approach for rebuilding materialized view

This script demonstrates the problematic full-recomputation approach where
the entire materialized view is rebuilt from scratch every time, regardless
of how much new data has arrived. This becomes increasingly slow as the
dataset grows over time.
"""

import pandas as pd
import pyarrow.parquet as pq
import glob
import json
import os
from datetime import datetime
from pathlib import Path


def rebuild_view():
    """
    Rebuild the materialized view from scratch by processing ALL transaction files.
    
    This is the SLOW approach - it reads and processes every transaction file
    every time, even if most of the data hasn't changed. Performance degrades
    linearly with total data volume.
    """
    try:
        print("Starting full view rebuild (slow approach)...")
        start_time = datetime.now()
        
        # Find ALL transaction files - this becomes problematic with large datasets
        transaction_files = glob.glob('/data/transactions/*.parquet')
        
        if not transaction_files:
            print("No transaction files found")
            return
        
        print(f"Found {len(transaction_files)} transaction files to process")
        
        # Load ALL transactions into memory - inefficient for large datasets
        all_transactions = []
        processed_dates = set()
        
        for file_path in transaction_files:
            print(f"Reading {file_path}...")
            df = pd.read_parquet(file_path)
            all_transactions.append(df)
            
            # Track which dates we've processed
            if 'date' in df.columns:
                processed_dates.update(df['date'].unique())
        
        # Concatenate all data - memory intensive
        print("Concatenating all transaction data...")
        full_df = pd.concat(all_transactions, ignore_index=True)
        
        # Perform aggregation on entire dataset - time consuming
        print("Computing aggregations on full dataset...")
        materialized_view = full_df.groupby(['product_id', 'date']).agg(
            total_sales=('amount', 'sum'),
            transaction_count=('amount', 'count')
        ).reset_index()
        
        # Ensure output directory exists
        output_dir = Path('/data/materialized_view')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the complete materialized view
        output_path = output_dir / 'view.parquet'
        print(f"Writing materialized view to {output_path}...")
        materialized_view.to_parquet(output_path, index=False)
        
        # Update metadata with all processed dates and timestamp
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'processed_dates': sorted([str(d) for d in processed_dates]),
            'total_files_processed': len(transaction_files),
            'total_records': len(materialized_view)
        }
        
        metadata_path = Path('/data/view_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"View rebuild complete!")
        print(f"Total records in view: {len(materialized_view)}")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"NOTE: This full rebuild approach scales poorly with data volume!")
        
    except Exception as e:
        print(f"Error during view rebuild: {e}")
        raise


if __name__ == '__main__':
    rebuild_view()