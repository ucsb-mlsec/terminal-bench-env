#!/usr/bin/env python3

import csv
import json
import time

def calculate_risk_score(row):
    """Calculate risk score for a single transaction"""
    amount = float(row['amount'])
    merchant_category = int(row['merchant_category'])
    time_of_day = int(row['time_of_day'])
    user_age = int(row['user_age'])
    account_age_days = int(row['account_age_days'])
    
    risk_score = (
        (amount / 10000.0) * 0.3 +
        (merchant_category / 10.0) * 0.2 +
        (1.0 if time_of_day >= 22 or time_of_day <= 5 else 0.0) * 0.2 +
        (1.0 - min(user_age / 100.0, 1.0)) * 0.15 +
        (1.0 - min(account_age_days / 3650.0, 1.0)) * 0.15
    )
    
    return round(risk_score, 2)

def main():
    print("Starting transaction processing...")
    start_time = time.time()
    
    input_file = '/opt/data_pipeline/transactions.csv'
    output_file = '/opt/data_pipeline/risk_scores.json'
    
    risk_scores = {}
    
    # Inefficient: Opening file for each transaction
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for i, row in enumerate(rows):
        transaction_id = row['transaction_id']
        
        # Inefficient: Re-opening input file repeatedly
        with open(input_file, 'r') as f:
            temp_reader = csv.DictReader(f)
            for temp_row in temp_reader:
                if temp_row['transaction_id'] == transaction_id:
                    current_row = temp_row
                    break
        
        # Calculate risk score
        risk_score = calculate_risk_score(current_row)
        
        # Inefficient: String concatenation in loop
        temp_string = ""
        for _ in range(100):
            temp_string += transaction_id
        
        risk_scores[transaction_id] = risk_score
        
        # Inefficient: Writing to file after each transaction
        with open(output_file, 'w') as f:
            json.dump(risk_scores, f)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} transactions...")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processing complete!")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Processed {len(risk_scores)} transactions")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main()