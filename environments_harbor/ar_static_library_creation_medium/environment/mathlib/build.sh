#!/bin/bash

# Build script for mathlib static library
# This script compiles all C source files into a static library

# Configuration
LIB_NAME="libmath.a"
OUTPUT_DIR="/workspace/mathlib"
SRC_DIR="/workspace/mathlib"

# Exit on any error
set -e

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Navigate to source directory
cd "$SRC_DIR"

# Find all C source files except the test program
echo "Finding source files..."
SOURCE_FILES=$(find . -maxdepth 1 -name "*.c" ! -name "test_math.c" -type f)

if [ -z "$SOURCE_FILES" ]; then
    echo "No source files found!"
    exit 1
fi

echo "Compiling source files..."
OBJECT_FILES=""

# Compile each source file into an object file
for src_file in $SOURCE_FILES; do
    obj_file="${src_file%.c}.o"
    echo "Compiling $src_file -> $obj_file"
    gcc -c -Wall -Wextra -O2 "$src_file" -o "$obj_file"
    OBJECT_FILES="$OBJECT_FILES $obj_file"
done

# Create static library from object files
echo "Creating static library $LIB_NAME..."
ar rcs "$OUTPUT_DIR/$LIB_NAME" $OBJECT_FILES

# Verify library was created
if [ ! -f "$OUTPUT_DIR/$LIB_NAME" ]; then
    echo "Error: Failed to create static library"
    exit 1
fi

echo "Static library created successfully: $OUTPUT_DIR/$LIB_NAME"

# List contents of library
echo "Library contents:"
ar -t "$OUTPUT_DIR/$LIB_NAME"

# Compile and link test program
if [ -f "test_math.c" ]; then
    echo "Compiling test program..."
    gcc -Wall -Wextra -O2 test_math.c -o test_math -L"$OUTPUT_DIR" -lmath -lm
    
    echo "Running test program..."
    ./test_math
    
    TEST_RESULT=$?
    if [ $TEST_RESULT -eq 0 ]; then
        echo "Test program executed successfully!"
    else
        echo "Test program failed with exit code: $TEST_RESULT"
        exit 1
    fi
else
    echo "Warning: test_math.c not found, skipping test"
fi

# Save solution path
echo "$OUTPUT_DIR/$LIB_NAME" > /workspace/solution.txt

echo "Build completed successfully!"
echo "Solution saved to /workspace/solution.txt"

# Cleanup object files
echo "Cleaning up object files..."
rm -f $OBJECT_FILES

echo "Done!"