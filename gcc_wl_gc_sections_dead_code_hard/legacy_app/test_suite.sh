#!/bin/bash

set -e

echo "Running test suite for legacy application..."

# Run the application and capture output
output=$(./app)
exit_code=$?

# Verify exit code is 0
if [ $exit_code -ne 0 ]; then
    echo "ERROR: Application exited with code $exit_code"
    exit 1
fi

# Check for expected output strings
if echo "$output" | grep -q "Core Function 1"; then
    echo "✓ Core Function 1 verified"
else
    echo "ERROR: Core Function 1 output missing"
    exit 1
fi

if echo "$output" | grep -q "Core Function 2"; then
    echo "✓ Core Function 2 verified"
else
    echo "ERROR: Core Function 2 output missing"
    exit 1
fi

echo "All tests passed"
exit 0