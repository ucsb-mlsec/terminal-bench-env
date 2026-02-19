#!/bin/bash

# Test script for legacy_app
# Validates that the compiled application functions correctly

set -e

echo "=== Legacy App Test Suite ==="

# Check if executable exists
if [ ! -f "./legacy_app" ]; then
    echo "ERROR: Executable ./legacy_app not found"
    exit 1
fi

echo "✓ Executable found"

# Make sure it's executable
chmod +x ./legacy_app

# Test 1: Self-test mode
echo "Running self-test..."
./legacy_app --self-test
if [ $? -ne 0 ]; then
    echo "ERROR: Self-test failed"
    exit 1
fi
echo "✓ Self-test passed"

# Test 2: Normal operation with no arguments
echo "Testing normal operation..."
./legacy_app > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Normal operation failed"
    exit 1
fi
echo "✓ Normal operation passed"

# Test 3: Help/version flag
echo "Testing help flag..."
./legacy_app --help > /dev/null 2>&1 || true
echo "✓ Help flag handled"

echo ""
echo "=== All Tests Passed ==="
exit 0