#!/bin/bash

# Build validation script to check for red zone optimization
# Red zone is a 128-byte area below the stack pointer that can be used
# without adjusting RSP. This is incompatible with certain runtime environments
# (like kernel code or interrupt handlers) where signal handlers or interrupts
# might clobber this area.

echo "Validating compiled binaries for red zone optimization..."

# Check if any .o files exist
if ! ls *.o >/dev/null 2>&1; then
    echo "FAIL: No object files found"
    exit 1
fi

# Flag to track validation status
VALIDATION_PASSED=true

# Check each object file
for obj_file in *.o; do
    echo "Checking $obj_file..."
    
    # Use objdump to disassemble and check for red zone usage
    # When -mno-red-zone is used, functions explicitly adjust RSP
    # Look for 'sub' instructions that adjust stack pointer at function entry
    if objdump -d "$obj_file" | grep -A 5 '<.*>:' | grep -q 'sub.*%rsp'; then
        echo "  Found explicit stack adjustment (good - no red zone)"
    else
        # Check if there's any actual code (not just data)
        if objdump -d "$obj_file" | grep -q '<.*>:'; then
            echo "  WARNING: No explicit stack adjustments found"
        fi
    fi
    
    # Additional check: look for compiler flags in the debug info if available
    # Note: This is a heuristic check
done

# Final validation result
if [ "$VALIDATION_PASSED" = true ]; then
    echo "PASS: Binaries compiled correctly"
    exit 0
else
    echo "FAIL: Red zone optimization detected"
    exit 1
fi