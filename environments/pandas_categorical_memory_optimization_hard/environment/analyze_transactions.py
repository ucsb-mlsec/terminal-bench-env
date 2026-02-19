#!/usr/bin/env python3
"""
E-commerce Transaction Analysis Script
This script loads and analyzes daily transaction data.
WARNING: This version is NOT memory optimized and consumes excessive RAM.
"""

import pandas as pd
import sys

def load_transaction_data(filepath):
    """
    Load transaction data from CSV file.
    No optimization applied - loads all columns as default dtypes.
    """
    print(f"Loading transaction data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def print_memory_usage(df):
    """Print detailed memory usage information."""
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    print(f"\nMemory Usage: {memory_mb:.2f} MB")
    print(f"Memory per column:")
    for col in df.columns:
        col_memory = df[col].memory_usage(deep=True) / (1024 * 1024)
        print(f"  {col}: {col_memory:.2f} MB")
    return memory_mb

def analyze_by_country(df):
    """Generate country-level sales report."""
    print("\n=== Analysis by Country ===")
    country_sales = df.groupby('country_code')['amount'].sum().sort_values(ascending=False)
    print(f"Top 5 countries by sales:\n{country_sales.head()}")
    return country_sales

def analyze_by_category(df):
    """Generate category-level transaction count report."""
    print("\n=== Analysis by Category ===")
    category_counts = df.groupby('category').size().sort_values(ascending=False)
    print(f"Transaction counts by category:\n{category_counts}")
    return category_counts

def analyze_by_payment_method(df):
    """Generate payment method average transaction report."""
    print("\n=== Analysis by Payment Method ===")
    payment_avg = df.groupby('payment_method')['amount'].mean().sort_values(ascending=False)
    print(f"Average transaction amount by payment method:\n{payment_avg}")
    return payment_avg

def analyze_by_status(df):
    """Generate status distribution report."""
    print("\n=== Analysis by Status ===")
    status_counts = df.groupby('status').size()
    print(f"Transaction counts by status:\n{status_counts}")
    return status_counts

def analyze_by_currency(df):
    """Generate currency-level totals report."""
    print("\n=== Analysis by Currency ===")
    currency_totals = df.groupby('currency')['amount'].sum().sort_values(ascending=False)
    print(f"Total amounts by currency:\n{currency_totals}")
    return currency_totals

def main():
    """Main execution function."""
    filepath = '/workspace/data/transactions.csv'
    
    # Load data without any optimization
    df = load_transaction_data(filepath)
    
    # Print basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Print memory usage
    memory_mb = print_memory_usage(df)
    
    # Perform analysis operations
    analyze_by_country(df)
    analyze_by_category(df)
    analyze_by_payment_method(df)
    analyze_by_status(df)
    analyze_by_currency(df)
    
    print(f"\n=== Summary ===")
    print(f"Total rows processed: {len(df)}")
    print(f"Total memory consumed: {memory_mb:.2f} MB")
    print("Analysis complete.")

if __name__ == "__main__":
    main()