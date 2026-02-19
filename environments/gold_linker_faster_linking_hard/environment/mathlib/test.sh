#!/bin/bash
set -e

CALCULATOR=./build/calculator

# Test addition: 2 + 2 = 4
result=$(${CALCULATOR} 2 + 2)
if [ "$result" != "4" ]; then
    echo "Addition test failed: expected 4, got $result"
    exit 1
fi

# Test subtraction: 10 - 3 = 7
result=$(${CALCULATOR} 10 - 3)
if [ "$result" != "7" ]; then
    echo "Subtraction test failed: expected 7, got $result"
    exit 1
fi

# Test multiplication: 6 * 7 = 42
result=$(${CALCULATOR} 6 \* 7)
if [ "$result" != "42" ]; then
    echo "Multiplication test failed: expected 42, got $result"
    exit 1
fi

# Test division: 15 / 3 = 5
result=$(${CALCULATOR} 15 / 3)
if [ "$result" != "5" ]; then
    echo "Division test failed: expected 5, got $result"
    exit 1
fi

echo "All tests passed!"
exit 0