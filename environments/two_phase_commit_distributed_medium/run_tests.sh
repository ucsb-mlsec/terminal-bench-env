#!/bin/bash

# Transaction Coordinator Test Script
# Tests the two-phase commit protocol implementation across multiple scenarios
# Ensures all participants reach consistent states (all COMMIT or all ABORT)

COORDINATOR_SCRIPT="/opt/coordinator/transaction_manager.py"
PASSED=0
TOTAL=5

echo "======================================"
echo "Transaction Coordinator Test Suite"
echo "======================================"
echo ""

# Test 1: All participants agree (vote YES) - should COMMIT on all
echo "Running Test 1: All participants agree"
OUTPUT=$(python3 $COORDINATOR_SCRIPT --test-scenario all_agree 2>&1)
EXIT_CODE=$?
if echo "$OUTPUT" | grep -q "COMMIT" && echo "$OUTPUT" | grep -q "consistent" && [ $EXIT_CODE -eq 0 ]; then
    echo "Test 1: PASS"
    ((PASSED++))
else
    echo "Test 1: FAIL"
    echo "Output: $OUTPUT"
fi
echo ""

# Test 2: One participant votes NO - should ABORT on all
echo "Running Test 2: One participant votes NO"
OUTPUT=$(python3 $COORDINATOR_SCRIPT --test-scenario one_no 2>&1)
EXIT_CODE=$?
if echo "$OUTPUT" | grep -q "ABORT" && echo "$OUTPUT" | grep -q "consistent" && [ $EXIT_CODE -eq 0 ]; then
    echo "Test 2: PASS"
    ((PASSED++))
else
    echo "Test 2: FAIL"
    echo "Output: $OUTPUT"
fi
echo ""

# Test 3: One participant times out - should ABORT on all
echo "Running Test 3: One participant times out"
OUTPUT=$(python3 $COORDINATOR_SCRIPT --test-scenario timeout 2>&1)
EXIT_CODE=$?
if echo "$OUTPUT" | grep -q "ABORT" && echo "$OUTPUT" | grep -q "consistent" && [ $EXIT_CODE -eq 0 ]; then
    echo "Test 3: PASS"
    ((PASSED++))
else
    echo "Test 3: FAIL"
    echo "Output: $OUTPUT"
fi
echo ""

# Test 4: Multiple participants vote NO - should ABORT on all
echo "Running Test 4: Multiple participants vote NO"
OUTPUT=$(python3 $COORDINATOR_SCRIPT --test-scenario multiple_no 2>&1)
EXIT_CODE=$?
if echo "$OUTPUT" | grep -q "ABORT" && echo "$OUTPUT" | grep -q "consistent" && [ $EXIT_CODE -eq 0 ]; then
    echo "Test 4: PASS"
    ((PASSED++))
else
    echo "Test 4: FAIL"
    echo "Output: $OUTPUT"
fi
echo ""

# Test 5: Mixed scenario with timeout and rejection - should ABORT on all
echo "Running Test 5: Mixed scenario with timeout and rejection"
OUTPUT=$(python3 $COORDINATOR_SCRIPT --test-scenario mixed 2>&1)
EXIT_CODE=$?
if echo "$OUTPUT" | grep -q "ABORT" && echo "$OUTPUT" | grep -q "consistent" && [ $EXIT_CODE -eq 0 ]; then
    echo "Test 5: PASS"
    ((PASSED++))
else
    echo "Test 5: FAIL"
    echo "Output: $OUTPUT"
fi
echo ""

# Print summary
echo "======================================"
echo "Test Summary"
echo "======================================"
echo "Total tests run: $TOTAL"
echo "Tests passed: $PASSED"

if [ $PASSED -eq $TOTAL ]; then
    echo "Overall result: SUCCESS"
    exit 0
else
    echo "Overall result: FAILURE"
    exit 1
fi