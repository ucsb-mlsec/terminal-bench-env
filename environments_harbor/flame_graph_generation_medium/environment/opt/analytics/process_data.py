#!/usr/bin/env python3

import json
import sys
import math
import time
from collections import defaultdict
from itertools import combinations

def load_data(filepath):
    """Load and parse JSON data from file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def calculate_statistics(numbers):
    """Calculate various statistics on a list of numbers"""
    if not numbers:
        return {}
    
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    
    stats = {
        'mean': sum(numbers) / n,
        'median': sorted_nums[n // 2] if n % 2 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2,
        'std_dev': 0,
        'min': min(numbers),
        'max': max(numbers)
    }
    
    # Calculate standard deviation with extra loops for CPU work
    mean = stats['mean']
    variance = sum((x - mean) ** 2 for x in numbers) / n
    stats['std_dev'] = math.sqrt(variance)
    
    return stats

def heavy_computation(record):
    """Perform CPU-intensive mathematical operations"""
    value = record.get('value', 0)
    name = record.get('name', '')
    
    # Heavy nested loop computation
    result = 0
    for i in range(1, 100):
        for j in range(1, 100):
            result += math.sin(value * i / 100) * math.cos(value * j / 100)
    
    # String operations
    processed_name = ''.join([c.upper() if i % 2 else c.lower() 
                              for i, c in enumerate(name * 10)])
    
    # More mathematical operations
    factorial_sum = sum(math.factorial(min(i, 10)) for i in range(20))
    
    return {
        'computation_result': result,
        'name_hash': hash(processed_name),
        'factorial_sum': factorial_sum,
        'original_value': value
    }

def medium_complexity_transform(records):
    """Transform and filter data with moderate complexity"""
    transformed = []
    
    for record in records:
        # Multiple transformations
        transformed_record = {
            'id': record.get('id', 0),
            'value_squared': record.get('value', 0) ** 2,
            'value_cubed': record.get('value', 0) ** 3,
            'name_length': len(record.get('name', '')),
            'category': categorize_value(record.get('value', 0))
        }
        
        # Nested list comprehension
        transformed_record['derived_values'] = [
            math.sqrt(abs(record.get('value', 0) * i)) 
            for i in range(1, 51)
        ]
        
        transformed.append(transformed_record)
    
    return transformed

def categorize_value(value):
    """Categorize a value into buckets"""
    if value < 100:
        return 'low'
    elif value < 500:
        return 'medium'
    elif value < 1000:
        return 'high'
    else:
        return 'very_high'

def find_patterns(records):
    """Find patterns and correlations in data"""
    patterns = defaultdict(list)
    
    # Group by category
    for record in records:
        category = record.get('category', 'unknown')
        patterns[category].append(record)
    
    # Calculate correlations
    correlations = {}
    for category, items in patterns.items():
        values = [item.get('value_squared', 0) for item in items]
        if values:
            correlations[category] = calculate_statistics(values)
    
    return correlations

def recursive_fibonacci(n):
    """Recursive Fibonacci for CPU load"""
    if n <= 1:
        return n
    return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)

def generate_combinations(records):
    """Generate combinations of records for analysis"""
    # Limit to first 15 records to avoid explosion
    limited_records = records[:15]
    
    results = []
    for combo in combinations(limited_records, 2):
        r1, r2 = combo
        v1 = r1.get('original_value', 0)
        v2 = r2.get('original_value', 0)
        
        results.append({
            'sum': v1 + v2,
            'product': v1 * v2,
            'difference': abs(v1 - v2)
        })
    
    return results

def aggregate_results(transformed_data, heavy_results, combinations):
    """Aggregate all processing results"""
    aggregated = {
        'total_records': len(transformed_data),
        'heavy_computation_sum': sum(r.get('computation_result', 0) for r in heavy_results),
        'average_value_squared': sum(r.get('value_squared', 0) for r in transformed_data) / max(len(transformed_data), 1),
        'total_combinations': len(combinations),
        'combination_sum': sum(c.get('sum', 0) for c in combinations)
    }
    
    # Additional processing
    all_derived = []
    for record in transformed_data:
        all_derived.extend(record.get('derived_values', []))
    
    if all_derived:
        aggregated['derived_statistics'] = calculate_statistics(all_derived)
    
    return aggregated

def process_data(data):
    """Main data processing pipeline"""
    records = data if isinstance(data, list) else data.get('records', [])
    
    print(f"Processing {len(records)} records...")
    
    # Stage 1: Heavy computation on each record
    heavy_results = []
    for record in records:
        result = heavy_computation(record)
        heavy_results.append(result)
    
    # Stage 2: Medium complexity transformations
    transformed_data = medium_complexity_transform(records)
    
    # Stage 3: Pattern finding
    patterns = find_patterns(transformed_data)
    
    # Stage 4: Generate combinations
    combinations_results = generate_combinations(heavy_results)
    
    # Stage 5: Recursive operations
    fib_results = [recursive_fibonacci(min(15 + i % 5, 20)) for i in range(10)]
    
    # Stage 6: Final aggregation
    final_results = aggregate_results(transformed_data, heavy_results, combinations_results)
    final_results['fibonacci_sum'] = sum(fib_results)
    final_results['patterns'] = patterns
    
    return final_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_data.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Load data
    data = load_data(input_file)
    
    # Process data
    results = process_data(data)
    
    # Print summary
    print(f"Processing complete!")
    print(f"Total records processed: {results.get('total_records', 0)}")
    print(f"Heavy computation sum: {results.get('heavy_computation_sum', 0):.2f}")
    print(f"Combinations analyzed: {results.get('total_combinations', 0)}")

if __name__ == '__main__':
    main()