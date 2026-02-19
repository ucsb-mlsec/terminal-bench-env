#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path

# Add current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from reducer import parallel_reduce
except ImportError as e:
    print(f"Error importing reducer module: {e}")
    sys.exit(1)

def load_test_data():
    """Load test cases from test_data.json"""
    test_file = Path(__file__).parent / 'test_data.json'
    try:
        with open(test_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: test_data.json not found at {test_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing test_data.json: {e}")
        sys.exit(1)

def run_tests():
    """Execute all test cases and report results"""
    test_data = load_test_data()
    
    if 'tests' not in test_data:
        print("Error: 'tests' key not found in test_data.json")
        sys.exit(1)
    
    tests = test_data['tests']
    passed = 0
    failed = 0
    
    for test_case in tests:
        name = test_case.get('name', 'Unknown')
        data = test_case.get('data', [])
        operation = test_case.get('operation', 'sum')
        expected = test_case.get('expected')
        
        try:
            result = parallel_reduce(data, operation)
            
            # Compare with tolerance for floating point operations
            tolerance = 1e-6
            if abs(result - expected) < tolerance:
                print(f"Test {name}: PASS")
                passed += 1
            else:
                print(f"Test {name}: FAIL (expected {expected}, got {result})")
                failed += 1
                
        except Exception as e:
            print(f"Test {name}: FAIL (exception: {e})")
            failed += 1
    
    # Print summary
    total = passed + failed
    print(f"\n{passed}/{total} tests passed")
    
    # Exit with appropriate code
    if failed == 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    run_tests()