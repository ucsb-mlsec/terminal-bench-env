#!/bin/bash

# Solution script to repair corrupted static library archive
# This script rebuilds the static library from the original object files
# and ensures proper symbol indexing

set -e  # Exit on any error

echo "Starting library repair process..."

# Step 1: Remove the corrupted library
echo "Removing corrupted library file..."
rm -f /workspace/lib/libcalc.a

# Step 2: Verify that object files exist
echo "Verifying object files..."
if [ ! -f /workspace/obj/add.o ] || [ ! -f /workspace/obj/subtract.o ] || \
   [ ! -f /workspace/obj/multiply.o ] || [ ! -f /workspace/obj/divide.o ]; then
    echo "Error: Required object files not found in /workspace/obj/"
    exit 1
fi

# Step 3: Rebuild the static library archive with proper symbol indexing
echo "Rebuilding static library from object files..."
ar rcs /workspace/lib/libcalc.a \
    /workspace/obj/add.o \
    /workspace/obj/subtract.o \
    /workspace/obj/multiply.o \
    /workspace/obj/divide.o

# Step 4: Verify the library was created successfully
if [ ! -f /workspace/lib/libcalc.a ]; then
    echo "Error: Failed to create library archive"
    exit 1
fi

# Step 5: Ensure proper symbol index (ranlib for extra assurance)
echo "Updating symbol index..."
ranlib /workspace/lib/libcalc.a

# Step 6: Verify library contents
echo "Verifying library contents..."
ar t /workspace/lib/libcalc.a

# Step 7: Test compilation and linking
echo "Testing compilation and linking..."
cd /workspace
gcc test_calc.c -L./lib -lcalc -o test_calc

# Step 8: Run the test program
echo "Running test program..."
./test_calc

echo "Library repair completed successfully!"