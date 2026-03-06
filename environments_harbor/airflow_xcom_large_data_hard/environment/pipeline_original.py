#!/usr/bin/env python3

import json
import os

# Stage 1: Load transaction data from file
def load_transactions():
    """Load all transactions from input file and return as list"""
    with open('/workspace/input/transactions.json', 'r') as f:
        transactions = json.load(f)
    return transactions  # PROBLEM: Returns large dataset in memory

# Stage 2: Enrich records with calculated fields
def enrich_records(transactions):
    """Add calculated fields to each transaction record"""
    enriched = []
    for txn in transactions:
        # Add amount category
        amount = txn['amount']
        if amount > 1000:
            txn['amount_category'] = 'high'
        elif amount > 100:
            txn['amount_category'] = 'medium'
        else:
            txn['amount_category'] = 'low'
        
        # Extract year and month from timestamp
        timestamp = txn['timestamp']
        txn['year'] = timestamp[:4]
        txn['month'] = timestamp[5:7]
        
        enriched.append(txn)
    
    return enriched  # PROBLEM: Returns large dataset in memory

# Stage 3: Aggregate data by category
def aggregate_by_category(enriched_transactions):
    """Aggregate transaction data by category"""
    aggregation = {}
    
    for txn in enriched_transactions:
        category = txn['category']
        if category not in aggregation:
            aggregation[category] = {
                'total_amount': 0,
                'count': 0
            }
        
        aggregation[category]['total_amount'] += txn['amount']
        aggregation[category]['count'] += 1
    
    return aggregation  # Still passing data in memory

# Stage 4: Generate summary statistics
def generate_summary(aggregated_data):
    """Calculate summary statistics from aggregated data"""
    total_records = sum(cat['count'] for cat in aggregated_data.values())
    total_amount = sum(cat['total_amount'] for cat in aggregated_data.values())
    category_count = len(aggregated_data)
    
    summary = {
        'total_records': total_records,
        'total_amount': total_amount,
        'category_count': category_count
    }
    
    return summary

if __name__ == "__main__":
    # Execute pipeline stages sequentially, passing data between them
    # This approach would fail with Airflow XCom for large datasets
    
    # Stage 1: Load data
    transactions = load_transactions()
    
    # Stage 2: Enrich records
    enriched_transactions = enrich_records(transactions)
    
    # Stage 3: Aggregate by category
    aggregated_data = aggregate_by_category(enriched_transactions)
    
    # Stage 4: Generate summary
    summary = generate_summary(aggregated_data)
    
    # Write output
    os.makedirs('/workspace/output', exist_ok=True)
    with open('/workspace/output/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Pipeline completed successfully")