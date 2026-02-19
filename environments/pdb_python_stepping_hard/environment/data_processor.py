#!/usr/bin/env python3

import csv
import sys

def process_data(filename):
    """
    Read CSV data and calculate statistics per category.
    Contains a subtle bug in the calculation logic.
    """
    results = {}
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                category = row['category']
                value = float(row['value'])
                
                # Initialize category if not seen before
                if category not in results:
                    results[category] = {
                        'count': 0,
                        'sum': 0.0,
                        'average': 0.0
                    }
                
                # BUG: Incrementing count before adding value
                # This causes the count to be 1 higher than actual values processed
                results[category]['count'] += 1
                
                # Add value to sum
                results[category]['sum'] += value
                
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data in CSV - {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing expected column - {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate averages
    for category in results:
        count = results[category]['count']
        total = results[category]['sum']
        if count > 0:
            results[category]['average'] = total / count
    
    return results

def print_report(results):
    """
    Print formatted report of the processing results.
    """
    print('=== Data Processing Report ===')
    print()
    
    # Sort categories alphabetically for consistent output
    sorted_categories = sorted(results.keys())
    
    for category in sorted_categories:
        stats = results[category]
        print(f"Category: {category}")
        print(f"  Count:   {stats['count']}")
        print(f"  Sum:     {stats['sum']:.2f}")
        print(f"  Average: {stats['average']:.2f}")
        print()
    
    # Print summary
    total_count = sum(r['count'] for r in results.values())
    total_sum = sum(r['sum'] for r in results.values())
    overall_avg = total_sum / total_count if total_count > 0 else 0.0
    
    print('--- Summary ---')
    print(f"Total records: {total_count}")
    print(f"Grand total:   {total_sum:.2f}")
    print(f"Overall avg:   {overall_avg:.2f}")

def main():
    """
    Main entry point for the data processor.
    """
    input_file = 'input_data.csv'
    results = process_data(input_file)
    print_report(results)

if __name__ == '__main__':
    main()