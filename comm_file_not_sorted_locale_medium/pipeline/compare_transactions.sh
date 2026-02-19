#!/bin/bash
# Transaction Comparison Pipeline
# Compares customer IDs from System A and System B

SYSTEM_A='/data/transactions/transactions_system_a.txt'
SYSTEM_B='/data/transactions/transactions_system_b.txt'

# Compare the two transaction files
comm "$SYSTEM_A" "$SYSTEM_B"