#!/bin/bash
# Library build script for libmolecular.so
# Originally created for GCC 9.4.0
# Compiles molecular dynamics force field calculation library

# Exit on any error
set -e

echo "Building libmolecular.so..."

# Compile the shared library
gcc -shared -fPIC -O2 libmolecular.c -o libmolecular.so -lm

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "Successfully compiled libmolecular.so"
else
    echo "Compilation failed!"
    exit 1
fi