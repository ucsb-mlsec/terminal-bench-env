#!/bin/bash

# Bash script to automate the testing process for the curve rasterization library

echo "===== Curve Rasterizer Test Suite ====="
echo ""

# Create output directory if it doesn't exist
echo "Step 1: Creating output directory..."
mkdir -p output
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create output directory"
    exit 1
fi
echo "Output directory ready."
echo ""

# Compile the rasterizer
echo "Step 2: Compiling rasterizer..."
gcc -o rasterizer rasterizer.c -lm -std=c99 -Wall
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "Compilation successful."
echo ""

# Run the rasterizer with test curves
echo "Step 3: Running rasterizer with test curves..."
./rasterizer test_curves.json output
if [ $? -ne 0 ]; then
    echo "ERROR: Rasterizer execution failed"
    exit 1
fi
echo "Rasterizer execution completed."
echo ""

# Run validation script
echo "Step 4: Validating output against references..."
python3 validate.py
VALIDATION_RESULT=$?
echo ""

# Display validation results
if [ $VALIDATION_RESULT -eq 0 ]; then
    echo "===== VALIDATION PASSED ====="
    echo "All test cases passed successfully!"
    exit 0
else
    echo "===== VALIDATION FAILED ====="
    echo "Some test cases did not pass validation."
    exit 1
fi