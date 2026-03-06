#!/usr/bin/env python3

from z3 import *

def create_problem():
    """Create a simple linear integer arithmetic problem."""
    solver = Solver()
    
    # Create integer variables
    x = Int('x')
    y = Int('y')
    z = Int('z')
    w = Int('w')
    v = Int('v')
    u = Int('u')
    t = Int('t')
    
    # Add linear constraints
    solver.add(x + y < 10)
    solver.add(x > 5)
    solver.add(y < 3)
    solver.add(z + w == 15)
    solver.add(w - v > 2)
    solver.add(u + t <= 20)
    solver.add(t >= 0)
    solver.add(v + x < 25)
    solver.add(z - y > 1)
    solver.add(u >= -5)
    solver.add(w + x + y <= 30)
    solver.add(t + v < 18)
    
    return solver