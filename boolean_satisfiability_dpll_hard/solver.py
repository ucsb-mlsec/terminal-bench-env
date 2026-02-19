#!/usr/bin/env python3

import os
import sys
import json

def parse_cnf(filename):
    """
    Parse DIMACS CNF file format.
    Returns: (num_vars, num_clauses, clauses)
    """
    num_vars = 0
    num_clauses = 0
    clauses = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('c'):
                continue
            
            # Parse problem line
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
                continue
            
            # Parse clause line
            # BUG: Not removing the trailing 0 properly
            clause = [int(x) for x in line.split()]
            clauses.append(clause)  # Should remove the 0 at the end!
    
    return num_vars, num_clauses, clauses

def unit_propagate(clauses, assignment):
    """
    Perform unit propagation on clauses given current assignment.
    Returns: (modified_clauses, new_assignments) or (None, None) if conflict
    """
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            # Filter out satisfied clauses and assigned literals
            unassigned = []
            satisfied = False
            
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                        satisfied = True
                        break
                else:
                    unassigned.append(lit)
            
            if satisfied:
                continue
            
            # Empty clause = conflict
            if len(unassigned) == 0:
                return None, None
            
            # Unit clause = propagate
            if len(unassigned) == 1:
                lit = unassigned[0]
                var = abs(lit)
                assignment[var] = (lit > 0)
                changed = True
    
    return clauses, assignment

def dpll(clauses, num_vars, assignment=None):
    """
    DPLL SAT solver algorithm.
    Returns: satisfying assignment dict or None if UNSAT
    """
    if assignment is None:
        assignment = {}
    
    # Unit propagation
    clauses, assignment = unit_propagate(clauses, assignment)
    if clauses is None:
        return None
    
    # Check if all clauses are satisfied
    all_satisfied = True
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                    clause_satisfied = True
                    break
        
        if not clause_satisfied:
            all_satisfied = False
            break
    
    if all_satisfied:
        return assignment
    
    # Find unassigned variable
    unassigned_var = None
    for i in range(1, num_vars + 1):
        if i not in assignment:
            unassigned_var = i
            break
    
    # BUG: Not checking if all variables are assigned but formula still not satisfied
    if unassigned_var is None:
        return None
    
    # Try assigning True
    new_assignment = assignment.copy()
    new_assignment[unassigned_var] = True
    result = dpll(clauses, num_vars, new_assignment)
    if result is not None:
        return result
    
    # Try assigning False
    # BUG: Should create a fresh copy from original assignment, not reuse
    new_assignment = assignment.copy()
    new_assignment[unassigned_var] = False
    result = dpll(clauses, num_vars, new_assignment)
    return result

def solve_sat(clauses, num_vars):
    """
    Solve SAT problem using DPLL algorithm.
    Returns: 'SAT' or 'UNSAT'
    """
    result = dpll(clauses, num_vars)
    if result is not None:
        return 'SAT'
    else:
        return 'UNSAT'

def main():
    """
    Main function to process all test cases and generate results.
    """
    # BUG: Wrong path - should be /workspace/test_cases/
    test_dir = '/workspace/test_case/'
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found")
        sys.exit(1)
    
    results = {}
    
    # Get all .cnf files
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.cnf')]
    test_files.sort()
    
    for test_file in test_files:
        filepath = os.path.join(test_dir, test_file)
        print(f"Processing {test_file}...")
        
        try:
            num_vars, num_clauses, clauses = parse_cnf(filepath)
            result = solve_sat(clauses, num_vars)
            results[test_file] = result
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error processing {test_file}: {e}")
            results[test_file] = "ERROR"
    
    # Save results to JSON
    output_file = '/workspace/results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()