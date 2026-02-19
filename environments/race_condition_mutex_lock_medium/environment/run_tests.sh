#!/bin/bash

# Payment Processor Consistency Test Script

echo "=== Payment Processor Consistency Test ==="
echo ""

# Compilation step
echo "Compiling payment_processor.c..."
gcc -pthread -o payment_processor payment_processor.c 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Arrays to store results
declare -a balances
declare -a transactions

# Run the program 5 times
echo "Running consistency tests..."
echo ""

for i in 1 2 3 4 5
do
    # Run the payment processor and capture output
    output=$(./payment_processor 2>&1)
    
    # Extract total balance and transactions from output
    balance=$(echo "$output" | grep -i "total balance" | grep -oE '[0-9]+' | head -1)
    trans=$(echo "$output" | grep -i "transactions processed" | grep -oE '[0-9]+' | head -1)
    
    # Store results
    balances[$i]=$balance
    transactions[$i]=$trans
    
    # Display run results
    echo "Run $i: Total Balance = $balance, Transactions = $trans"
done

echo ""
echo "=== Results Summary ==="

# Check consistency
consistent="YES"
first_balance=${balances[1]}
first_trans=${transactions[1]}

for i in 2 3 4 5
do
    if [ "${balances[$i]}" != "$first_balance" ] || [ "${transactions[$i]}" != "$first_trans" ]; then
        consistent="NO"
        break
    fi
done

echo "CONSISTENT: $consistent"
echo "Expected total: 10000"
echo ""

if [ "$consistent" = "YES" ]; then
    echo "✓ All runs produced identical results"
else
    echo "✗ Inconsistent results detected - race conditions present"
fi