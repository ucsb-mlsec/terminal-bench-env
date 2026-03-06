#!/usr/bin/env python3

import pandas as pd
import json
import os
from pathlib import Path

def main():
    try:
        # Read cleaned transactions data
        input_file = 'data/cleaned_transactions.csv'
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        df = pd.read_csv(input_file)
        
        # Calculate metrics
        total_transactions = int(len(df))
        average_amount = round(float(df['amount'].mean()), 2)
        total_revenue = round(float(df['amount'].sum()), 2)
        
        # Prepare metrics dictionary
        metrics = {
            'total_transactions': total_transactions,
            'average_amount': average_amount,
            'total_revenue': total_revenue
        }
        
        # Create output directory if it doesn't exist
        output_file = 'data/metrics.json'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON file
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print('Metrics calculated successfully.')
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except KeyError as e:
        print(f"Error: Missing required column {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == '__main__':
    main()