#!/usr/bin/env python3
"""
Broken optimization script - DOES NOT properly constrain atoms
This script allows ALL atoms to move during optimization, which violates
the requirement that atoms 0, 1, 2, 3 should remain fixed.
"""

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS

# Define initial atomic positions (Angstroms)
# Atoms 0-3 should be FIXED but currently are NOT constrained
positions = [
    [0.0, 0.0, 0.0],    # atom 0 - should be FIXED but isn't
    [2.0, 0.0, 0.0],    # atom 1 - should be FIXED but isn't
    [0.0, 2.0, 0.0],    # atom 2 - should be FIXED but isn't
    [2.0, 2.0, 0.0],    # atom 3 - should be FIXED but isn't
    [1.0, 1.0, 1.5],    # atom 4 - MOVABLE
    [1.0, 1.0, 3.0],    # atom 5 - MOVABLE
    [0.5, 0.5, 2.0],    # atom 6 - MOVABLE
    [1.5, 0.5, 2.0],    # atom 7 - MOVABLE
    [0.5, 1.5, 2.0],    # atom 8 - MOVABLE
    [1.5, 1.5, 2.0],    # atom 9 - MOVABLE
    [1.0, 0.5, 2.5],    # atom 10 - MOVABLE
    [1.0, 1.5, 2.5],    # atom 11 - MOVABLE
]

# Create atoms object - all Hydrogen atoms
atoms = Atoms('H12', positions=positions)

# BUG: Missing constraint application here!
# Should use FixAtoms to constrain atoms 0, 1, 2, 3
# Without constraints, all atoms will move during optimization

# Attach EMT calculator
atoms.calc = EMT()

# Set up BFGS optimizer
# fmax=0.01 eV/Angstrom convergence criterion
# Maximum 200 steps
optimizer = BFGS(atoms, trajectory=None)

print("Starting optimization (BROKEN - all atoms can move)...")
print("WARNING: Atoms 0-3 should be fixed but are not constrained!")

# Run optimization
optimizer.run(fmax=0.01, steps=200)

print("Optimization complete (but results are incorrect due to missing constraints)")
print("\nFinal positions:")
for i, pos in enumerate(atoms.get_positions()):
    print(f"Atom {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")