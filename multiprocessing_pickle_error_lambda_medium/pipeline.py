#!/usr/bin/env python3

import csv
import json
import os
from multiprocessing import Pool


class TransactionProcessor:
    def __init__(self, discount_threshold=100, discount_rate=0.05):
        self.discount_threshold = discount_threshold
        self.discount_rate = discount_rate
    
    def apply_discount(self, transaction):
        """Apply discount to transactions over threshold"""
        amount = float(transaction['amount'])
        if amount > self.discount_threshold:
            amount = amount * (1 - self.discount_rate)
        transaction['amount'] = amount
        return transaction
    
    def categorize_transaction(self, transaction):
        """Categorize transaction based on amount"""
        amount = float(transaction['amount'])
        if amount < 50:
            transaction['category'] = 'small'
        elif amount < 200:
            transaction['category'] = 'medium'
        else:
            transaction['category'] = 'large'
        return transaction
    
    def validate_transaction(self, transaction):
        """Validate and mark transaction as processed"""
        transaction['processed'] = True
        transaction['amount'] = float(transaction['amount'])
        return transaction
    
    def process_transaction(self, transaction):
        """Apply all transformations to a transaction"""
        transaction = self.validate_transaction(transaction)
        transaction = self.apply_discount(transaction)
        transaction = self.categorize_transaction(transaction)
        return transaction


def main():
    # Setup paths
    input_file = '/workspace/data/transactions.csv'
    output_file = '/workspace/output/results.json'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read transaction data
    transactions = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append(row)
    
    # Create processor instance
    processor = TransactionProcessor()
    
    # Process transactions in parallel using multiprocessing
    # This will fail with PicklingError because instance methods cannot be serialized
    with Pool(processes=4) as pool:
        processed_transactions = pool.map(processor.process_transaction, transactions)
    
    # Aggregate results
    total_processed = len(processed_transactions)
    total_amount = sum(float(t['amount']) for t in processed_transactions)
    
    # Save results
    results = {
        "total_processed": total_processed,
        "total_amount": total_amount,
        "status": "success"
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Successfully processed {total_processed} transactions")
    print(f"Total amount: {total_amount}")
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()