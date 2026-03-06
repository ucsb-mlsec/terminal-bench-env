#!/usr/bin/env python3

from z3 import *

def create_problem():
    """
    Creates a Z3 solver with linear integer arithmetic and boolean constraints.
    This is a FAST workload designed to solve in under 2 seconds.
    """
    solver = Solver()
    
    # Create 5 integer variables
    x = Int('x')
    y = Int('y')
    z = Int('z')
    w = Int('w')
    v = Int('v')
    
    # Create 5 boolean variables
    b1 = Bool('b1')
    b2 = Bool('b2')
    b3 = Bool('b3')
    b4 = Bool('b4')
    b5 = Bool('b5')
    
    # Add linear integer arithmetic constraints with conditionals
    solver.add(If(b1, x > 10, x < 5))
    solver.add(If(b2, y + z < 20, y + z > 30))
    solver.add(If(b3, w - v > 0, w - v < -5))
    
    # Add boolean implications with linear arithmetic
    solver.add(Implies(b1, y + z < 50))
    solver.add(Implies(b2, x + w > 0))
    solver.add(Implies(b3, v < 100))
    
    # Add some direct linear constraints
    solver.add(x + y + z < 100)
    solver.add(w + v > -10)
    solver.add(x - y < 20)
    
    # Add boolean constraints
    solver.add(Or(b1, b2, b3))
    solver.add(Implies(b4, And(b1, b2)))
    solver.add(Implies(b5, Or(b3, b4)))
    
    # Add range constraints
    solver.add(x >= -50)
    solver.add(x <= 50)
    solver.add(y >= 0)
    solver.add(z <= 100)
    
    return solver