#!/usr/bin/env python3

import numpy as np
import compute_core

def test_sum_array():
    """Test the sum_array function"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_core.sum_array(data)
    expected = 15.0
    if abs(result - expected) < 1e-10:
        print('Test 1: PASS')
        return True
    else:
        print('Test 1: FAIL')
        return False

def test_multiply_arrays():
    """Test the multiply_arrays function"""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    result = compute_core.multiply_arrays(a, b)
    expected = np.array([2.0, 6.0, 12.0, 20.0])
    if np.allclose(result, expected):
        print('Test 2: PASS')
        return True
    else:
        print('Test 2: FAIL')
        return False

def test_compute_mean():
    """Test the compute_mean function"""
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = compute_core.compute_mean(data)
    expected = 30.0
    if abs(result - expected) < 1e-10:
        print('Test 3: PASS')
        return True
    else:
        print('Test 3: FAIL')
        return False

if __name__ == '__main__':
    passed = 0
    
    if test_sum_array():
        passed += 1
    
    if test_multiply_arrays():
        passed += 1
    
    if test_compute_mean():
        passed += 1
    
    print(f'Summary: {passed}/3 tests passed')