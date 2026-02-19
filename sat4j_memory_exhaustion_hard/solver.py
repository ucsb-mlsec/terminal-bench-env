#!/usr/bin/env python3

import json
import sys
import argparse
import os
from pysat.solvers import Glucose3
from pysat.formula import CNF

def load_config(config_path='config.json'):
    """
    Load solver configuration from JSON file.
    These parameters control memory usage and solver behavior.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file '{config_path}'.")
        sys.exit(1)

def solve_cnf(cnf_file, config):
    """
    Solve a CNF problem using PySAT with the given configuration.
    
    Parameters in config affect memory usage:
    - propagate_limit: Controls propagation budget (None = unlimited, causes high memory)
    - clause_limit: Controls learned clause database size (None = unlimited)
    - use_phase_saving: Phase saving can increase memory usage
    - preprocessing: Aggressive preprocessing can bloat memory
    """
    
    # Load CNF formula from DIMACS file
    try:
        formula = CNF(from_file=cnf_file)
    except FileNotFoundError:
        print(f"Error: CNF file '{cnf_file}' not found.")
        sys.exit(1)
    
    print(f"Loaded CNF with {formula.nv} variables and {len(formula.clauses)} clauses")
    
    # Initialize solver with configuration parameters
    # These parameters are intentionally set to cause high memory usage in default config
    solver = Glucose3()
    
    # Apply configuration settings that affect memory
    # Note: PySAT's interface is limited, but we simulate configuration impact
    # In real scenarios, these would be set during solver initialization
    
    # Set propagation and clause limits if specified
    # Default (unset) values cause unlimited storage and high memory usage
    if 'propagate_limit' in config and config['propagate_limit'] is not None:
        # Limited propagation budget reduces memory
        pass  # PySAT doesn't expose this directly, but config simulates it
    
    if 'clause_limit' in config and config['clause_limit'] is not None:
        # Limited clause database reduces memory
        pass  # PySAT doesn't expose this directly
    
    # Add all clauses from the formula to the solver
    for clause in formula.clauses:
        solver.add_clause(clause)
    
    print("Solving...")
    
    # Solve the problem
    result = solver.solve()
    
    if result:
        print("Result: SATISFIABLE")
        # Get the model (variable assignments)
        model = solver.get_model()
        if model and len(model) <= 50:  # Only print small models
            print(f"Model: {model[:20]}..." if len(model) > 20 else f"Model: {model}")
        return "SAT", model
    else:
        print("Result: UNSATISFIABLE")
        return "UNSAT", None
    
    solver.delete()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SAT Solver Service using PySAT')
    parser.add_argument('cnf_file', nargs='?', default='test_problem.cnf',
                       help='Path to CNF file in DIMACS format (default: test_problem.cnf)')
    parser.add_argument('--config', default='config.json',
                       help='Path to configuration file (default: config.json)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration: {config}")
    
    # Solve the CNF problem
    result, model = solve_cnf(args.cnf_file, config)
    
    return 0 if result == "SAT" else 1

if __name__ == '__main__':
    sys.exit(main())