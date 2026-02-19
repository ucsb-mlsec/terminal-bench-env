import os
import pandas as pd
from dagster import Definitions, asset, AssetExecutionContext


@asset
def process_sales_data(context: AssetExecutionContext) -> str:
    """
    Process sales transaction data and generate summary report.
    Reads CSV data, calculates total revenue per product, and saves as JSON.
    """
    # Read input data
    input_path = '/workspace/sales_pipeline/data/transactions.csv'
    context.log.info(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Calculate total revenue per product (quantity * price)
    df['revenue'] = df['quantity'] * df['price']
    summary = df.groupby('product')['revenue'].sum().reset_index()
    summary.columns = ['product', 'total_revenue']
    
    # Prepare output
    output_path = '/workspace/sales_pipeline/output/summary.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON with records format
    records = summary.to_dict('records')
    summary.to_json(output_path, orient='records', indent=2)
    
    context.log.info(f"Summary saved to {output_path}")
    context.log.info(f"Processed {len(records)} products")
    
    return output_path


defs = Definitions(
    assets=[process_sales_data]
)