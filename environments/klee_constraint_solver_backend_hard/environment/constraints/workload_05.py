#!/usr/bin/env python3

from z3 import *

def create_problem():
    """
    Create a Z3 solver with array theory constraints.
    Uses simple select/store operations with linear constraints.
    This is designed to be a FAST workload.
    """
    solver = Solver()
    
    # Create array variables: arrays mapping integers to integers
    arr1 = Array('arr1', IntSort(), IntSort())
    arr2 = Array('arr2', IntSort(), IntSort())
    arr3 = Array('arr3', IntSort(), IntSort())
    
    # Create index and value variables
    i = Int('i')
    j = Int('j')
    k = Int('k')
    v1 = Int('v1')
    v2 = Int('v2')
    v3 = Int('v3')
    
    # Add linear constraints on indices
    solver.add(i >= 0)
    solver.add(i < 10)
    solver.add(j >= 0)
    solver.add(j < 10)
    solver.add(k >= 0)
    solver.add(k < 10)
    solver.add(i != j)
    
    # Store operations: create modified arrays
    arr1_modified = Store(arr1, i, v1)
    arr2_modified = Store(arr2, j, v2)
    
    # Select operations: read values from arrays
    solver.add(Select(arr1_modified, i) == v1)
    solver.add(Select(arr2_modified, j) == v2)
    solver.add(Select(arr3, k) == v3)
    
    # Linear constraints on values
    solver.add(v1 >= 1)
    solver.add(v1 <= 100)
    solver.add(v2 >= 1)
    solver.add(v2 <= 100)
    solver.add(v3 >= 1)
    solver.add(v3 <= 100)
    solver.add(v1 + v2 == v3)
    
    # Additional array constraint: values at different indices
    solver.add(Select(arr1, j) != Select(arr2, i))
    
    # Store and select chain
    arr_chain = Store(Store(arr3, i, 10), j, 20)
    solver.add(Select(arr_chain, i) == 10)
    solver.add(Select(arr_chain, j) == 20)
    
    return solver