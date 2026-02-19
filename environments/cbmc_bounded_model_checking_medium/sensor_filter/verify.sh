#!/bin/bash

# Read UNWIND_BOUND from verify_config.txt
if [ ! -f "verify_config.txt" ]; then
    echo "Error: verify_config.txt not found"
    exit 1
fi

BOUND=$(grep "UNWIND_BOUND" verify_config.txt | awk -F'=' '{print $2}' | tr -d ' ')

if [ -z "$BOUND" ]; then
    echo "Error: UNWIND_BOUND not found in config file"
    exit 1
fi

echo "Running verification with UNWIND_BOUND=$BOUND..."
echo "Executing: cbmc filter.c --unwind $BOUND --unwinding-assertions"
echo ""

# Simulate CBMC verification behavior based on bound value
if [ "$BOUND" -lt 10 ]; then
    echo "WARNING: Unwinding incomplete for loop iterations"
    echo "Some execution paths were not fully explored"
    echo "VERIFICATION INCOMPLETE"
    echo ""
    echo "** Analysis terminated: Cannot guarantee completeness **"
    echo "Increase UNWIND_BOUND to explore deeper execution paths"
    exit 2
elif [ "$BOUND" -ge 10 ] && [ "$BOUND" -lt 12 ]; then
    echo "Unwinding complete - all paths explored"
    echo ""
    echo "Checking array bounds..."
    echo "ERROR: Array index out of bounds detected"
    echo "  File: filter.c, Line: 45"
    echo "  Condition: buffer[index] where index=10 exceeds array size=10"
    echo ""
    echo "VERIFICATION FAILED - Safety violation found"
    exit 1
else
    echo "Unwinding complete - all paths explored"
    echo ""
    echo "Checking array bounds... PASSED"
    echo "Checking pointer safety... PASSED"
    echo "Checking assertions... PASSED"
    echo "Checking arithmetic overflow... PASSED"
    echo ""
    echo "VERIFICATION SUCCESSFUL - No violations found"
    exit 0
fi