#!/usr/bin/env python3

from prefect import flow, task
import pandas as pd
import os

@task
def read_sales_data(file_path):
    """Read sales data from CSV file"""
    print(f"Reading sales data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    # Bug 1: Not returning the dataframe
    # return df

@task
def calculate_statistics(data):
    """Calculate statistics from sales data"""
    print("Calculating statistics...")
    
    # Bug 2: Using wrong variable name (data instead of df)
    total_records = len(df)
    
    # Calculate total revenue
    total_revenue = data['revenue'].sum()
    
    # Bug 3: Division by zero when calculating average
    average_revenue = total_revenue / 0
    
    stats = {
        'records_processed': total_records,
        'total_revenue': round(total_revenue, 2),
        'average_revenue': round(average_revenue, 2)
    }
    
    return stats

@task
def write_results(stats, output_path):
    """Write results to output file"""
    print(f"Writing results to {output_path}")
    
    status = "COMPLETED" if stats else "FAILED"
    
    with open(output_path, 'w') as f:
        f.write(f"status={status}\n")
        f.write(f"records_processed={stats['records_processed']}\n")
        f.write(f"total_revenue={stats['total_revenue']}\n")
    
    print("Results written successfully")

@flow(name="sales-processing-flow")
def process_sales_flow():
    """Main flow to process sales data"""
    print("Starting sales processing flow...")
    
    input_file = "/workspace/sales_data.csv"
    output_file = "/workspace/result.txt"
    
    # Read the sales data
    sales_data = read_sales_data(input_file)
    
    # Calculate statistics
    stats = calculate_statistics(sales_data)
    
    # Write results
    write_results(stats, output_file)
    
    print("Flow completed successfully!")
    return stats

if __name__ == "__main__":
    process_sales_flow()