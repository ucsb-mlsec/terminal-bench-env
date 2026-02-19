#!/usr/bin/env python3

from z3 import *

def create_problem():
    """
    Create a Z3 solver with linear real arithmetic constraints.
    This workload uses only linear constraints which Z3 can solve efficiently.
    """
    solver = Solver()
    
    # Create 8 real variables
    x1 = Real('x1')
    x2 = Real('x2')
    x3 = Real('x3')
    x4 = Real('x4')
    x5 = Real('x5')
    x6 = Real('x6')
    x7 = Real('x7')
    x8 = Real('x8')
    
    # Add linear real arithmetic constraints
    solver.add(2.5 * x1 + 3.1 * x2 < 10.0)
    solver.add(x1 >= 0.0)
    solver.add(x2 >= 0.0)
    solver.add(1.5 * x3 + 2.0 * x4 + 0.5 * x5 <= 15.0)
    solver.add(x3 + x4 >= 5.0)
    solver.add(x5 - x6 == 2.0)
    solver.add(x6 >= 1.0)
    solver.add(x6 <= 10.0)
    solver.add(3.0 * x7 + 1.2 * x8 >= 7.5)
    solver.add(x7 + x8 <= 20.0)
    solver.add(x1 + x2 + x3 >= 3.0)
    solver.add(x4 + x5 + x6 + x7 <= 50.0)
    solver.add(2.0 * x1 - 1.5 * x3 + x7 >= 0.0)
    solver.add(x8 >= 0.0)
    solver.add(x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 <= 100.0)
    
    return solver