#!/usr/bin/env python3

from z3 import *

def create_problem():
    """
    Creates a Z3 solver with bitvector constraints using small bit-widths.
    This workload uses 8-bit bitvectors with standard operations.
    Should be FAST (under 2 seconds).
    """
    solver = Solver()
    
    # Create 7 bitvector variables with 8-bit width
    x = BitVec('x', 8)
    y = BitVec('y', 8)
    z = BitVec('z', 8)
    a = BitVec('a', 8)
    b = BitVec('b', 8)
    c = BitVec('c', 16)
    d = BitVec('d', 16)
    
    # Add bitvector constraints using various operations
    # Addition constraints
    solver.add(x + y == 100)
    solver.add(z + a == 50)
    
    # Bitwise AND constraints
    solver.add((x & y) == 32)
    solver.add((a & b) == 15)
    
    # Bitwise OR constraints
    solver.add((x | z) == 255)
    solver.add((y | a) == 127)
    
    # Bitwise XOR constraints
    solver.add((x ^ y) == 68)
    solver.add((z ^ b) == 45)
    
    # Mixed operations with 16-bit variables
    solver.add(c + d == 1000)
    solver.add((c & d) == 256)
    solver.add((c | d) == 1500)
    
    # Additional constraints combining operations
    solver.add(x > 10)
    solver.add(y < 200)
    solver.add(z != 0)
    solver.add(a > b)
    
    return solver