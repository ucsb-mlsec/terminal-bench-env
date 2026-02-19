import pandas as pd


def calculate_total_sales(sales_df: pd.DataFrame) -> float:
    """Calculate total sales amount.
    
    Args:
        sales_df: DataFrame containing sales transactions with 'amount' column
        
    Returns:
        Total sum of all sales amounts as a float
    """
    return float(sales_df['amount'].sum())


def count_unique_customers(sales_df: pd.DataFrame) -> int:
    """Count unique customers.
    
    Args:
        sales_df: DataFrame containing sales transactions with 'customer_id' column
        
    Returns:
        Count of unique customer IDs as an integer
    """
    return int(sales_df['customer_id'].nunique())