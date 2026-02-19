#!/usr/bin/env python3

from z3 import *

def create_problem():
    """
    Creates a Z3 solver with nonlinear arithmetic constraints.
    This workload should be SLOW due to complex nonlinear operations.
    """
    solver = Solver()
    
    # Create integer variables
    x = Int('x')
    y = Int('y')
    z = Int('z')
    a = Int('a')
    b = Int('b')
    c = Int('c')
    d = Int('d')
    
    # Add nonlinear arithmetic constraints
    # Polynomial constraints
    solver.add(x*x + y*y == 169)
    solver.add(x*x*x + y*y == 50)
    solver.add(z*z*z - x*x == 27)
    
    # Modulo operations (nonlinear)
    solver.add(a % b == c)
    solver.add(b > 0)
    solver.add(a > 10)
    solver.add(c >= 0)
    solver.add(c < 10)
    
    # More complex nonlinear constraints
    solver.add(x*y*z == 120)
    solver.add(a*a + b*b + c*c == 100)
    solver.add(d*d == a + b + c)
    
    # Additional constraints mixing linear and nonlinear
    solver.add(x + y + z == 20)
    solver.add(a*b == 36)
    solver.add(c*d*d == 48)
    
    # Bounds to keep search space finite but still challenging
    solver.add(x >= -20)
    solver.add(x <= 20)
    solver.add(y >= -20)
    solver.add(y <= 20)
    solver.add(z >= -20)
    solver.add(z <= 20)
    solver.add(d >= 0)
    solver.add(d <= 20)
    
    return solver