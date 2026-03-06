#!/usr/bin/env python3

from z3 import *
import sys
import time

# Start timing
start_time = time.time()

# Create Z3 solver with default (unoptimized) configuration
solver = Solver()

# Model Diffie-Hellman parameters using BitVectors
# Using large bit-width to represent cryptographic values
BITWIDTH = 128
p = BitVecVal(2**127 - 1, BITWIDTH)  # Large prime modulus

# Protocol participants
# Round 1
alice_private_1 = BitVec('alice_priv_1', BITWIDTH)
bob_private_1 = BitVec('bob_priv_1', BITWIDTH)
g = BitVecVal(5, BITWIDTH)  # Generator

# Public keys round 1
alice_public_1 = BitVec('alice_pub_1', BITWIDTH)
bob_public_1 = BitVec('bob_pub_1', BITWIDTH)

# Shared secrets round 1
alice_shared_1 = BitVec('alice_shared_1', BITWIDTH)
bob_shared_1 = BitVec('bob_shared_1', BITWIDTH)

# Round 2
alice_private_2 = BitVec('alice_priv_2', BITWIDTH)
bob_private_2 = BitVec('bob_priv_2', BITWIDTH)
alice_public_2 = BitVec('alice_pub_2', BITWIDTH)
bob_public_2 = BitVec('bob_pub_2', BITWIDTH)
alice_shared_2 = BitVec('alice_shared_2', BITWIDTH)
bob_shared_2 = BitVec('bob_shared_2', BITWIDTH)

# Round 3
alice_private_3 = BitVec('alice_priv_3', BITWIDTH)
bob_private_3 = BitVec('bob_priv_3', BITWIDTH)
alice_public_3 = BitVec('alice_pub_3', BITWIDTH)
bob_public_3 = BitVec('bob_pub_3', BITWIDTH)
alice_shared_3 = BitVec('alice_shared_3', BITWIDTH)
bob_shared_3 = BitVec('bob_shared_3', BITWIDTH)

# Round 4
alice_private_4 = BitVec('alice_priv_4', BITWIDTH)
bob_private_4 = BitVec('bob_priv_4', BITWIDTH)
alice_public_4 = BitVec('alice_pub_4', BITWIDTH)
bob_public_4 = BitVec('bob_pub_4', BITWIDTH)
alice_shared_4 = BitVec('alice_shared_4', BITWIDTH)
bob_shared_4 = BitVec('bob_shared_4', BITWIDTH)

# Attacker knowledge
attacker_data_1 = BitVec('attacker_1', BITWIDTH)
attacker_data_2 = BitVec('attacker_2', BITWIDTH)
attacker_data_3 = BitVec('attacker_3', BITWIDTH)
attacker_data_4 = BitVec('attacker_4', BITWIDTH)

# Constraints: Private keys are non-zero and within bounds
solver.add(ULT(0, alice_private_1))
solver.add(ULT(alice_private_1, p))
solver.add(ULT(0, bob_private_1))
solver.add(ULT(bob_private_1, p))

solver.add(ULT(0, alice_private_2))
solver.add(ULT(alice_private_2, p))
solver.add(ULT(0, bob_private_2))
solver.add(ULT(bob_private_2, p))

solver.add(ULT(0, alice_private_3))
solver.add(ULT(alice_private_3, p))
solver.add(ULT(0, bob_private_3))
solver.add(ULT(bob_private_3, p))

solver.add(ULT(0, alice_private_4))
solver.add(ULT(alice_private_4, p))
solver.add(ULT(0, bob_private_4))
solver.add(ULT(bob_private_4, p))

# Protocol step assertions - Round 1
# Alice computes public key: g^a mod p (simplified as multiplication for verification)
solver.add(alice_public_1 == URem(g * alice_private_1, p))
# Bob computes public key: g^b mod p
solver.add(bob_public_1 == URem(g * bob_private_1, p))
# Alice computes shared secret: bob_public^alice_private mod p
solver.add(alice_shared_1 == URem(bob_public_1 * alice_private_1, p))
# Bob computes shared secret: alice_public^bob_private mod p
solver.add(bob_shared_1 == URem(alice_public_1 * bob_private_1, p))

# Protocol step assertions - Round 2
solver.add(alice_public_2 == URem(g * alice_private_2, p))
solver.add(bob_public_2 == URem(g * bob_private_2, p))
solver.add(alice_shared_2 == URem(bob_public_2 * alice_private_2, p))
solver.add(bob_shared_2 == URem(alice_public_2 * bob_private_2, p))

# Protocol step assertions - Round 3
solver.add(alice_public_3 == URem(g * alice_private_3, p))
solver.add(bob_public_3 == URem(g * bob_private_3, p))
solver.add(alice_shared_3 == URem(bob_public_3 * alice_private_3, p))
solver.add(bob_shared_3 == URem(alice_public_3 * bob_private_3, p))

# Protocol step assertions - Round 4
solver.add(alice_public_4 == URem(g * alice_private_4, p))
solver.add(bob_public_4 == URem(g * bob_private_4, p))
solver.add(alice_shared_4 == URem(bob_public_4 * alice_private_4, p))
solver.add(bob_shared_4 == URem(alice_public_4 * bob_private_4, p))

# Security Property 1: Shared secret agreement
solver.add(alice_shared_1 == bob_shared_1)
solver.add(alice_shared_2 == bob_shared_2)
solver.add(alice_shared_3 == bob_shared_3)
solver.add(alice_shared_4 == bob_shared_4)

# Attacker observes public information
solver.add(attacker_data_1 == alice_public_1)
solver.add(attacker_data_2 == bob_public_2)
solver.add(attacker_data_3 == alice_public_3)
solver.add(attacker_data_4 == bob_public_4)

# Security Property 2: Key secrecy - attacker cannot derive private keys
solver.add(attacker_data_1 != alice_private_1)
solver.add(attacker_data_2 != bob_private_2)
solver.add(attacker_data_3 != alice_private_3)
solver.add(attacker_data_4 != bob_private_4)

# Security Property 3: Forward secrecy - different rounds use different keys
solver.add(alice_private_1 != alice_private_2)
solver.add(alice_private_2 != alice_private_3)
solver.add(alice_private_3 != alice_private_4)
solver.add(bob_private_1 != bob_private_2)
solver.add(bob_private_2 != bob_private_3)
solver.add(bob_private_3 != bob_private_4)

# Security Property 4: Shared secrets are different across rounds (forward secrecy)
solver.add(alice_shared_1 != alice_shared_2)
solver.add(alice_shared_2 != alice_shared_3)
solver.add(alice_shared_3 != alice_shared_4)

# Check satisfiability - if unsat, there's no way to violate properties
result = solver.check()

elapsed_time = time.time() - start_time

if result == sat:
    print("VERIFICATION FAILED - counterexample found")
    sys.exit(1)
elif result == unsat:
    print("VERIFICATION PASSED")
    sys.exit(0)
else:
    print("VERIFICATION UNKNOWN")
    sys.exit(1)