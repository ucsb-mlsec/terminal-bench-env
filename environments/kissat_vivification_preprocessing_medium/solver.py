#!/usr/bin/env python3

import sys
import os

class FastSAT:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.clauses = []
        self.num_vars = 0
        self.preprocessed_stats = {}
        
    def load_config(self, config_path):
        """Load configuration from file"""
        config = {}
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) == 2:
                            option_name = parts[0]
                            value = int(parts[1])
                            config[option_name] = value
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
        return config
    
    def parse_cnf(self, filename):
        """Parse DIMACS CNF file"""
        clauses = []
        num_vars = 0
        num_clauses = 0
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    parts = line.split()
                    num_vars = int(parts[2])
                    num_clauses = int(parts[3])
                    continue
                
                # Parse clause
                literals = [int(x) for x in line.split() if int(x) != 0]
                if literals:
                    clauses.append(literals)
        
        self.clauses = clauses
        self.num_vars = num_vars
        return clauses
    
    def unit_propagation(self, clauses):
        """Apply unit propagation"""
        if not self.config.get('unit_propagation', 0):
            return clauses, 0
        
        changes = 0
        unit_clauses = [c for c in clauses if len(c) == 1]
        
        while unit_clauses:
            unit = unit_clauses[0][0]
            new_clauses = []
            
            for clause in clauses:
                if unit in clause:
                    # Clause is satisfied
                    changes += 1
                    continue
                elif -unit in clause:
                    # Remove the literal
                    new_clause = [lit for lit in clause if lit != -unit]
                    if new_clause:
                        new_clauses.append(new_clause)
                    changes += 1
                else:
                    new_clauses.append(clause)
            
            clauses = new_clauses
            unit_clauses = [c for c in clauses if len(c) == 1]
        
        return clauses, changes
    
    def pure_literal_elimination(self, clauses):
        """Apply pure literal elimination"""
        if not self.config.get('pure_literal', 0):
            return clauses, 0
        
        changes = 0
        literals = {}
        
        for clause in clauses:
            for lit in clause:
                if lit not in literals:
                    literals[lit] = True
        
        pure_literals = []
        for lit in literals:
            if -lit not in literals:
                pure_literals.append(lit)
        
        if pure_literals:
            new_clauses = []
            for clause in clauses:
                satisfied = False
                for lit in pure_literals:
                    if lit in clause:
                        satisfied = True
                        changes += 1
                        break
                if not satisfied:
                    new_clauses.append(clause)
            clauses = new_clauses
        
        return clauses, changes
    
    def subsumption(self, clauses):
        """Apply clause subsumption"""
        if not self.config.get('subsumption', 0):
            return clauses, 0
        
        changes = 0
        new_clauses = []
        
        for i, clause1 in enumerate(clauses):
            subsumed = False
            for j, clause2 in enumerate(clauses):
                if i != j and set(clause2).issubset(set(clause1)):
                    subsumed = True
                    changes += 1
                    break
            if not subsumed:
                new_clauses.append(clause1)
        
        return new_clauses, changes
    
    def vivification(self, clauses):
        """Apply vivification preprocessing"""
        if not self.config.get('vivify', 0):
            return clauses, 0
        
        changes = 0
        new_clauses = []
        
        for clause in clauses:
            if len(clause) <= 1:
                new_clauses.append(clause)
                continue
            
            # Try to remove literals from the clause
            simplified = clause[:]
            for lit in clause:
                test_clause = [l for l in simplified if l != lit]
                if test_clause and len(test_clause) < len(simplified):
                    # Simplified version
                    simplified = test_clause
                    changes += 1
            
            new_clauses.append(simplified)
        
        return new_clauses, changes
    
    def preprocess(self):
        """Apply all enabled preprocessing techniques"""
        print("\n=== Preprocessing Phase ===")
        original_count = len(self.clauses)
        
        # Vivification
        vivify_enabled = self.config.get('vivify', 0)
        self.clauses, vivify_changes = self.vivification(self.clauses)
        print(f"Preprocessing applied: vivification - {'YES' if vivify_enabled else 'NO'}")
        if vivify_enabled:
            print(f"  Vivification changes: {vivify_changes}")
        
        # Unit propagation
        up_enabled = self.config.get('unit_propagation', 0)
        self.clauses, up_changes = self.unit_propagation(self.clauses)
        print(f"Preprocessing applied: unit_propagation - {'YES' if up_enabled else 'NO'}")
        if up_enabled:
            print(f"  Unit propagation changes: {up_changes}")
        
        # Pure literal elimination
        pl_enabled = self.config.get('pure_literal', 0)
        self.clauses, pl_changes = self.pure_literal_elimination(self.clauses)
        print(f"Preprocessing applied: pure_literal - {'YES' if pl_enabled else 'NO'}")
        if pl_enabled:
            print(f"  Pure literal changes: {pl_changes}")
        
        # Subsumption
        sub_enabled = self.config.get('subsumption', 0)
        self.clauses, sub_changes = self.subsumption(self.clauses)
        print(f"Preprocessing applied: subsumption - {'YES' if sub_enabled else 'NO'}")
        if sub_enabled:
            print(f"  Subsumption changes: {sub_changes}")
        
        final_count = len(self.clauses)
        print(f"\nClauses reduced: {original_count} -> {final_count}")
        print("=== Preprocessing Complete ===\n")
    
    def dpll_solve(self, clauses, assignment):
        """Simple DPLL solver"""
        # Check if all clauses satisfied
        if not clauses:
            return True, assignment
        
        # Check for empty clause
        if any(len(c) == 0 for c in clauses):
            return False, None
        
        # Unit propagation
        unit_clauses = [c for c in clauses if len(c) == 1]
        if unit_clauses:
            unit = unit_clauses[0][0]
            new_assignment = assignment.copy()
            new_assignment[abs(unit)] = (unit > 0)
            
            new_clauses = []
            for clause in clauses:
                if unit in clause:
                    continue
                elif -unit in clause:
                    new_clause = [lit for lit in clause if lit != -unit]
                    if not new_clause:
                        return False, None
                    new_clauses.append(new_clause)
                else:
                    new_clauses.append(clause)
            
            return self.dpll_solve(new_clauses, new_assignment)
        
        # Choose a variable
        var = abs(clauses[0][0])
        
        # Try positive assignment
        new_assignment = assignment.copy()
        new_assignment[var] = True
        new_clauses = []
        for clause in clauses:
            if var in clause:
                continue
            elif -var in clause:
                new_clause = [lit for lit in clause if lit != -var]
                if not new_clause:
                    pass
                else:
                    new_clauses.append(new_clause)
            else:
                new_clauses.append(clause)
        
        sat, result = self.dpll_solve(new_clauses, new_assignment)
        if sat:
            return True, result
        
        # Try negative assignment
        new_assignment = assignment.copy()
        new_assignment[var] = False
        new_clauses = []
        for clause in clauses:
            if -var in clause:
                continue
            elif var in clause:
                new_clause = [lit for lit in clause if lit != var]
                if not new_clause:
                    pass
                else:
                    new_clauses.append(new_clause)
            else:
                new_clauses.append(clause)
        
        return self.dpll_solve(new_clauses, new_assignment)
    
    def solve(self, cnf_file):
        """Main solve routine"""
        print(f"FastSAT Solver")
        print(f"Loading CNF file: {cnf_file}")
        
        self.parse_cnf(cnf_file)
        print(f"Parsed {len(self.clauses)} clauses with {self.num_vars} variables")
        
        self.preprocess()
        
        print("=== Solving Phase ===")
        sat, assignment = self.dpll_solve(self.clauses, {})
        
        if sat:
            print("Result: SATISFIABLE")
        else:
            print("Result: UNSATISFIABLE")
        
        return sat

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 solver.py <cnf_file>")
        sys.exit(1)
    
    cnf_file = sys.argv[1]
    
    # Get config path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.txt')
    
    solver = FastSAT(config_path)
    solver.solve(cnf_file)

if __name__ == "__main__":
    main()