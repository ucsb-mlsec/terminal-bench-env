#!/bin/bash

# Test Harness for Transaction Service
# Tests correctness under high concurrency

set -e

echo "=== Transaction Service Test Harness ==="
echo

# Configuration
NUM_ACCOUNTS=10
INITIAL_BALANCE=1000
EXPECTED_TOTAL=$((NUM_ACCOUNTS * INITIAL_BALANCE))
NUM_ITERATIONS=5
TEST_EXECUTABLE="./transaction_service"

# Clean and compile
echo "Step 1: Cleaning and compiling..."
cd /workspace/transaction_service
make clean > /dev/null 2>&1
if ! make; then
    echo "FAIL: Compilation failed"
    exit 1
fi
echo "Compilation successful"
echo

# Run stress tests
PASS_COUNT=0
FAIL_COUNT=0

for i in $(seq 1 $NUM_ITERATIONS); do
    echo "Step 2: Running stress test iteration $i/$NUM_ITERATIONS..."
    
    # Run the transaction service in stress test mode
    if ! $TEST_EXECUTABLE --stress > test_output_$i.txt 2>&1; then
        echo "FAIL: Iteration $i crashed (exit code: $?)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Check if output file exists
    if [ ! -f test_output_$i.txt ]; then
        echo "FAIL: Iteration $i - no output file generated"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Extract final total balance from output
    FINAL_TOTAL=$(grep -oP "Final total balance: \K\d+" test_output_$i.txt || echo "0")
    TRANSACTIONS_COMPLETED=$(grep -oP "Transactions completed: \K\d+" test_output_$i.txt || echo "0")
    
    echo "  Transactions completed: $TRANSACTIONS_COMPLETED"
    echo "  Final total balance: $FINAL_TOTAL"
    echo "  Expected total balance: $EXPECTED_TOTAL"
    
    # Validate results
    if [ "$FINAL_TOTAL" -eq "$EXPECTED_TOTAL" ] && [ "$TRANSACTIONS_COMPLETED" -eq "1000" ]; then
        echo "  Result: PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  Result: FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo
done

# Summary
echo "=== Test Summary ==="
echo "Passed: $PASS_COUNT/$NUM_ITERATIONS"
echo "Failed: $FAIL_COUNT/$NUM_ITERATIONS"
echo

# Final verdict
if [ $PASS_COUNT -eq $NUM_ITERATIONS ]; then
    echo "OVERALL: PASS"
    echo "All tests passed successfully!"
    exit 0
else
    echo "OVERALL: FAIL"
    echo "Some tests failed. Check output above for details."
    exit 1
fi