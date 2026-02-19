from z3 import *

def create_problem():
    """
    Creates a boolean SAT problem with propositional logic constraints.
    This is a FAST workload that should solve quickly.
    """
    solver = Solver()
    
    # Create 10 boolean variables
    b1 = Bool('b1')
    b2 = Bool('b2')
    b3 = Bool('b3')
    b4 = Bool('b4')
    b5 = Bool('b5')
    b6 = Bool('b6')
    b7 = Bool('b7')
    b8 = Bool('b8')
    b9 = Bool('b9')
    b10 = Bool('b10')
    
    # Add propositional logic constraints
    solver.add(Or(b1, b2, b3))
    solver.add(Implies(b1, And(b4, b5)))
    solver.add(Or(Not(b2), b6))
    solver.add(And(Or(b3, b7), Or(b4, b8)))
    solver.add(Implies(b5, Not(b6)))
    solver.add(Or(b7, Not(b8), b9))
    solver.add(And(Implies(b9, b10), Implies(b10, Not(b1))))
    solver.add(Or(b2, b3, Not(b4)))
    solver.add(And(Or(b5, b6), Or(b7, b8)))
    solver.add(Implies(And(b9, b10), Or(b1, b2)))
    
    return solver