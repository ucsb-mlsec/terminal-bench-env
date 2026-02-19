#!/usr/bin/env python3

import json
import os
import sys
import time


def load_json(filepath):
    """Load and return JSON data from a file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        raise


def compute_optimal_cost(dimensions):
    """
    Compute the minimum number of scalar multiplications needed for matrix chain multiplication.
    Uses dynamic programming approach.
    
    Args:
        dimensions: List where dimensions[i] and dimensions[i+1] define the i-th matrix
    
    Returns:
        Minimum number of scalar multiplications (integer)
    """
    n = len(dimensions) - 1  # Number of matrices
    
    if n <= 1:
        return 0
    
    # m[i][j] represents minimum cost to multiply matrices from i to j
    m = [[0] * n for _ in range(n)]
    
    # l is the chain length
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = float('inf')
            
            # Try all possible split points
            for k in range(i, j):
                # Cost of multiplying matrices i to k, k+1 to j, and then the two results
                cost = m[i][k] + m[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                
                if cost < m[i][j]:
                    m[i][j] = cost
    
    return int(m[0][n - 1])


def validate_parenthesization(paren_str, num_matrices):
    """
    Validate that a parenthesization string is well-formed.
    
    Args:
        paren_str: The parenthesization string to validate
        num_matrices: Expected number of matrices
    
    Returns:
        True if valid, False otherwise
    """
    if not paren_str:
        return False
    
    # Check that all matrices A1 through A{num_matrices} are present
    for i in range(1, num_matrices + 1):
        if f"A{i}" not in paren_str:
            return False
    
    # Check balanced parentheses
    balance = 0
    for char in paren_str:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        if balance < 0:
            return False
    
    if balance != 0:
        return False
    
    # Check that ' x ' is used as operator
    if num_matrices > 1 and ' x ' not in paren_str:
        return False
    
    return True


def validate_solution():
    """
    Validate the solution against the input test cases.
    
    Returns:
        True if validation passes, False otherwise
    """
    start_time = time.time()
    
    # Load input and solution files
    try:
        input_data = load_json('/workspace/matrix_chains.json')
        solution_data = load_json('/workspace/solution.json')
    except Exception as e:
        print(f"Failed to load files: {e}")
        return False
    
    # Check solution structure
    if 'results' not in solution_data:
        print("Error: solution.json must contain 'results' array")
        return False
    
    results = solution_data['results']
    
    # Check that we have the same number of results as test cases
    if len(results) != len(input_data):
        print(f"Error: Expected {len(input_data)} results, got {len(results)}")
        return False
    
    # Validate each test case
    all_passed = True
    for idx, test_case in enumerate(input_data):
        test_id = test_case['id']
        dimensions = test_case['dimensions']
        num_matrices = len(dimensions) - 1
        
        # Find corresponding result
        result = results[idx]
        
        # Check ID matches
        if result['id'] != test_id:
            print(f"Error: Test case {idx}: ID mismatch. Expected '{test_id}', got '{result['id']}'")
            all_passed = False
            continue
        
        # Compute optimal cost
        expected_cost = compute_optimal_cost(dimensions)
        actual_cost = result['min_operations']
        
        # Validate min_operations
        if actual_cost != expected_cost:
            print(f"Error: Test case '{test_id}': min_operations mismatch.")
            print(f"  Expected: {expected_cost}")
            print(f"  Got: {actual_cost}")
            all_passed = False
            continue
        
        # Validate parenthesization string
        paren_str = result['parenthesization']
        if not validate_parenthesization(paren_str, num_matrices):
            print(f"Error: Test case '{test_id}': Invalid parenthesization string.")
            print(f"  Got: '{paren_str}'")
            print(f"  Expected {num_matrices} matrices (A1 through A{num_matrices})")
            all_passed = False
            continue
        
        # Success for this test case
        print(f"✓ Test case '{test_id}' passed (cost: {actual_cost})")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal validation time: {elapsed_time:.2f} seconds")
    
    if elapsed_time > 60:
        print("Warning: Solution took longer than expected (>60 seconds)")
    
    return all_passed


if __name__ == '__main__':
    print("Starting validation...\n")
    
    if validate_solution():
        print("\n" + "=" * 50)
        print("VALIDATION PASSED")
        print("=" * 50)
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("VALIDATION FAILED")
        print("=" * 50)
        sys.exit(1)