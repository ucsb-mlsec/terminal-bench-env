#!/usr/bin/env python3

import pycosat
import os

def read_cnf(filename):
    """Read a CNF file in DIMACS format and return clauses."""
    clauses = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('c'):
                continue
            # Skip the problem line
            if line.startswith('p'):
                continue
            # Parse clause
            literals = list(map(int, line.split()))
            if literals[-1] == 0:
                literals = literals[:-1]
            if literals:
                clauses.append(literals)
    return clauses

def read_patterns(filename):
    """Read patterns from file, one per line."""
    patterns = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                literals = list(map(int, line.split()))
                patterns.append(literals)
    return patterns

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read circuit constraints
    cnf_file = os.path.join(script_dir, 'circuit.cnf')
    circuit_clauses = read_cnf(cnf_file)
    
    # Read patterns to test
    patterns_file = os.path.join(script_dir, 'patterns.txt')
    patterns = read_patterns(patterns_file)
    
    # BUG: We're creating a single list and adding patterns as permanent clauses
    # This causes contamination between pattern checks
    all_clauses = circuit_clauses.copy()
    
    # Check each pattern
    for i, pattern in enumerate(patterns, 1):
        # BUG: Adding each literal from the pattern as a unit clause permanently
        # This should use assumptions instead!
        for literal in pattern:
            all_clauses.append([literal])
        
        # Check satisfiability
        result = pycosat.solve(all_clauses)
        
        if result == "UNSAT" or result == "UNSATISFIABLE":
            print(f"Pattern {i}: UNSAT")
        else:
            print(f"Pattern {i}: SAT")

if __name__ == "__main__":
    main()