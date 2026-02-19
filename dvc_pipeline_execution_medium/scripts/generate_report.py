#!/usr/bin/env python3

import json
import os
import sys

def generate_report():
    """Generate final report from calculated metrics."""
    try:
        # Read metrics from intermediate file
        metrics_path = 'data/metrics.json'
        if not os.path.exists(metrics_path):
            print(f"Error: Metrics file not found at {metrics_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Transform data to final report format
        final_report = {
            'transactions': int(metrics['total_transactions']),
            'avg_amount': float(metrics['average_amount']),
            'total_revenue': float(metrics['total_revenue'])
        }
        
        # Ensure output directory exists
        output_path = 'data/final_report.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write final report
        with open(output_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print('Final report generated successfully.')
        
    except KeyError as e:
        print(f"Error: Missing expected key in metrics file: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid value format in metrics: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in metrics file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    generate_report()