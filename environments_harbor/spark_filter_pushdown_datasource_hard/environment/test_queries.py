#!/usr/bin/env python3

import time
import sys
import os
from reader import CSVReader

def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"  Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def test_equality_filter():
    """Test query with single equality filter - should only process matching rows"""
    print("\n=== Test 1: Equality Filter (id = 5000) ===")
    reader = CSVReader('/workspace/data/sensors.csv')
    
    # Create filter predicate for id = 5000
    filters = {'id': ('equals', 5000)}
    results = reader.read_with_filters(filters)
    
    print(f"  Results returned: {len(results)}")
    print(f"  Rows processed by reader: {reader.rows_processed}")
    if len(results) > 0:
        print(f"  Sample result: {results[0]}")
    return results

@timer
def test_range_filter():
    """Test query with range conditions - should skip rows outside range"""
    print("\n=== Test 2: Range Filter (timestamp > 1000 AND timestamp < 2000) ===")
    reader = CSVReader('/workspace/data/sensors.csv')
    
    # Create range filter
    filters = {
        'timestamp': ('range', 1000, 2000)
    }
    results = reader.read_with_filters(filters)
    
    print(f"  Results returned: {len(results)}")
    print(f"  Rows processed by reader: {reader.rows_processed}")
    if len(results) > 0:
        print(f"  Sample result: {results[0]}")
    return results

@timer
def test_comparison_filter():
    """Test query with comparison operator - should filter during read"""
    print("\n=== Test 3: Comparison Filter (temperature > 75.0) ===")
    reader = CSVReader('/workspace/data/sensors.csv')
    
    # Create comparison filter
    filters = {'temperature': ('greater_than', 75.0)}
    results = reader.read_with_filters(filters)
    
    print(f"  Results returned: {len(results)}")
    print(f"  Rows processed by reader: {reader.rows_processed}")
    if len(results) > 0:
        print(f"  Sample result: {results[0]}")
    return results

@timer
def test_multiple_filters():
    """Test query with multiple combined filters - should apply all during read"""
    print("\n=== Test 4: Multiple Filters (id > 1000 AND temperature < 50.0) ===")
    reader = CSVReader('/workspace/data/sensors.csv')
    
    # Create multiple filters
    filters = {
        'id': ('greater_than', 1000),
        'temperature': ('less_than', 50.0)
    }
    results = reader.read_with_filters(filters)
    
    print(f"  Results returned: {len(results)}")
    print(f"  Rows processed by reader: {reader.rows_processed}")
    if len(results) > 0:
        print(f"  Sample result: {results[0]}")
    return results

@timer
def test_no_filter():
    """Test query without filters - should read all data (baseline)"""
    print("\n=== Test 5: No Filter (SELECT * FROM sensors) ===")
    reader = CSVReader('/workspace/data/sensors.csv')
    
    # No filters
    results = reader.read_with_filters(None)
    
    print(f"  Results returned: {len(results)}")
    print(f"  Rows processed by reader: {reader.rows_processed}")
    if len(results) > 0:
        print(f"  Sample result: {results[0]}")
    return results

def main():
    """Run all test queries and display performance metrics"""
    print("=" * 60)
    print("CSV Reader Filter Pushdown Performance Tests")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists('/workspace/data/sensors.csv'):
        print("ERROR: Data file not found at /workspace/data/sensors.csv")
        sys.exit(1)
    
    try:
        # Run all tests
        test_equality_filter()
        test_range_filter()
        test_comparison_filter()
        test_multiple_filters()
        test_no_filter()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()