#!/bin/bash

set -e

echo "Running test suite..."

# Run the legacy_app and capture output
OUTPUT=$(./legacy_app)
EXIT_CODE=$?

# Check exit status
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: Program exited with status $EXIT_CODE"
    exit 1
fi

# Verify expected output strings
echo "$OUTPUT" | grep -q "Result:" || { echo "FAIL: Missing 'Result:' in output"; exit 1; }
echo "$OUTPUT" | grep -q "Sum:" || { echo "FAIL: Missing 'Sum:' in output"; exit 1; }
echo "$OUTPUT" | grep -q "Maximum:" || { echo "FAIL: Missing 'Maximum:' in output"; exit 1; }

# Additional validation - check for specific values
echo "$OUTPUT" | grep -q "Sum: 15" || { echo "FAIL: Incorrect sum value"; exit 1; }
echo "$OUTPUT" | grep -q "Maximum: 5" || { echo "FAIL: Incorrect maximum value"; exit 1; }

echo "PASS: All tests passed successfully"
echo "Program output:"
echo "$OUTPUT"

exit 0