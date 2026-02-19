"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""

import pandas as pd
from typing import Dict


def calculate_revenue(sales_data: pd.DataFrame) -> float:
    """
    Calculate total revenue from sales data.
    
    Args:
        sales_data: DataFrame with columns product_id, quantity, and price
        
    Returns:
        Total revenue as a float (sum of quantity * price for all rows)
    """
    sales_data['revenue'] = sales_data['quantity'] * sales_data['price']
    total_revenue = sales_data['revenue'].sum()
    return float(total_revenue)


def save_revenue(revenue: float) -> Dict[str, float]:
    """
    Save revenue value in a dictionary format.
    
    Args:
        revenue: Total revenue value as a float
        
    Returns:
        Dictionary with key 'total_revenue' and the revenue value
    """
    return {'total_revenue': revenue}