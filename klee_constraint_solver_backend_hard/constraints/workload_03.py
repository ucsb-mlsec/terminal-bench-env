from z3 import *

def create_problem():
    """
    Creates a Z3 solver instance with nonlinear arithmetic constraints.
    This is a SLOW workload due to the complexity of nonlinear arithmetic.
    """
    solver = Solver()
    
    # Create integer and real variables
    x = Int('x')
    y = Int('y')
    z = Int('z')
    a = Real('a')
    b = Real('b')
    c = Real('c')
    d = Real('d')
    e = Int('e')
    
    # Add nonlinear arithmetic constraints
    # Multiplication constraints
    solver.add(x * y == z)
    solver.add(a * b * c > 100)
    solver.add(x * y * e == 1000)
    
    # More complex nonlinear relationships
    solver.add(a * a + b * b == c * c)  # Pythagorean-like constraint
    solver.add(x * x - y * y == 144)
    solver.add(a * b == d * d)
    
    # Division constraints (nonlinear)
    solver.add(c * d == 50)
    solver.add(a > 0)
    solver.add(b > 0)
    solver.add(d > 0)
    
    # Additional nonlinear constraints to increase complexity
    solver.add(x * e > 50)
    solver.add(y * z < 10000)
    solver.add(b * c * d < 500)
    
    # Range constraints
    solver.add(x > 0)
    solver.add(y > 0)
    solver.add(z > 0)
    solver.add(e > 1)
    solver.add(c > 0)
    
    # More nonlinear interactions
    solver.add(x * y * z > 100)
    solver.add(a * b + b * c + c * a == 200)
    
    return solver