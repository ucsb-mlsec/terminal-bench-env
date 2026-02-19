#!/usr/bin/env python3

import pandas as pd
import dask.dataframe as dd
import dask
from datetime import datetime

def create_sample_data():
    """Create sample transaction data"""
    data = {
        'transaction_id': range(1, 10001),
        'customer_id': [f'CUST_{i % 1000}' for i in range(1, 10001)],
        'date': pd.date_range('2023-01-01', periods=10000, freq='H'),
        'amount': [100 + (i % 500) for i in range(10000)],
        'category': [['Electronics', 'Clothing', 'Food', 'Books'][i % 4] for i in range(10000)],
        'customer_segment': [['Premium', 'Standard', 'Basic'][i % 3] for i in range(10000)]
    }
    df = pd.DataFrame(data)
    return dd.from_pandas(df, npartitions=4)

def create_pipeline():
    """
    Create analytics pipeline with intentional redundant filtering operations
    """
    # Load base data
    base_data = create_sample_data()
    
    # REDUNDANT DATE RANGE FILTERS - Applied 3 times separately
    # Filter 1: For total sum calculation
    date_filter_1 = base_data[
        (base_data['date'] >= '2023-01-01') & 
        (base_data['date'] <= '2023-12-31')
    ]
    total_amount_sum = date_filter_1['amount'].sum()
    
    # Filter 2: For mean calculation (same date filter!)
    date_filter_2 = base_data[
        (base_data['date'] >= '2023-01-01') & 
        (base_data['date'] <= '2023-12-31')
    ]
    total_amount_mean = date_filter_2['amount'].mean()
    
    # Filter 3: For count calculation (same date filter again!)
    date_filter_3 = base_data[
        (base_data['date'] >= '2023-01-01') & 
        (base_data['date'] <= '2023-12-31')
    ]
    total_transaction_count = date_filter_3['transaction_id'].count()
    
    # REDUNDANT CUSTOMER SEGMENT FILTERS - Applied 3 times separately
    # Premium segment filter 1: For sum
    premium_filter_1 = base_data[base_data['customer_segment'] == 'Premium']
    premium_sum = premium_filter_1['amount'].sum()
    
    # Premium segment filter 2: For mean (same filter!)
    premium_filter_2 = base_data[base_data['customer_segment'] == 'Premium']
    premium_mean = premium_filter_2['amount'].mean()
    
    # Premium segment filter 3: For count (same filter again!)
    premium_filter_3 = base_data[base_data['customer_segment'] == 'Premium']
    premium_count = premium_filter_3['transaction_id'].count()
    
    # REDUNDANT CATEGORY FILTERS - Applied 2 times separately
    # Electronics filter 1: For sum
    electronics_filter_1 = base_data[base_data['category'] == 'Electronics']
    electronics_sum = electronics_filter_1['amount'].sum()
    
    # Electronics filter 2: For mean (same filter!)
    electronics_filter_2 = base_data[base_data['category'] == 'Electronics']
    electronics_mean = electronics_filter_2['amount'].mean()
    
    # COMBINED REDUNDANT FILTERS - Applied 2 times
    # Combined filter 1: Date + Premium
    combined_filter_1 = base_data[
        (base_data['date'] >= '2023-01-01') & 
        (base_data['date'] <= '2023-12-31') &
        (base_data['customer_segment'] == 'Premium')
    ]
    combined_sum_1 = combined_filter_1['amount'].sum()
    
    # Combined filter 2: Date + Premium (same filter!)
    combined_filter_2 = base_data[
        (base_data['date'] >= '2023-01-01') & 
        (base_data['date'] <= '2023-12-31') &
        (base_data['customer_segment'] == 'Premium')
    ]
    combined_mean_1 = combined_filter_2['amount'].mean()
    
    # Standard segment filters - Applied 2 times
    standard_filter_1 = base_data[base_data['customer_segment'] == 'Standard']
    standard_sum = standard_filter_1['amount'].sum()
    
    standard_filter_2 = base_data[base_data['customer_segment'] == 'Standard']
    standard_count = standard_filter_2['transaction_id'].count()
    
    # Collect all computations
    results = {
        'total_sum': total_amount_sum,
        'total_mean': total_amount_mean,
        'total_count': total_transaction_count,
        'premium_sum': premium_sum,
        'premium_mean': premium_mean,
        'premium_count': premium_count,
        'electronics_sum': electronics_sum,
        'electronics_mean': electronics_mean,
        'combined_sum': combined_sum_1,
        'combined_mean': combined_mean_1,
        'standard_sum': standard_sum,
        'standard_count': standard_count
    }
    
    return results

def get_task_graph():
    """
    Get the task graph from the pipeline
    """
    results = create_pipeline()
    
    # Create a combined delayed computation to capture all tasks
    all_computations = [v for v in results.values()]
    
    # Get the task graph
    graph = dask.base.collections_to_dsk(all_computations)
    
    return dict(graph)

if __name__ == '__main__':
    print("Creating analytics pipeline with redundant filters...")
    results = create_pipeline()
    
    print("\nGetting task graph...")
    graph = get_task_graph()
    
    print(f"\nTotal tasks in graph: {len(graph)}")
    print("\nPipeline created successfully!")
    print("The task graph contains multiple redundant filtering operations.")