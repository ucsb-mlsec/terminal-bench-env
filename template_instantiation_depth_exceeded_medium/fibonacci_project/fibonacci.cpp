#!/bin/bash

# Fix template instantiation depth error and build the Fibonacci project

# Change to project directory
cd /workspace/fibonacci_project

# Modify the Makefile to add compiler flag for increased template depth
# The flag -ftemplate-depth=N increases the maximum template instantiation depth
sed -i 's/CXXFLAGS = /CXXFLAGS = -ftemplate-depth=2048 /' Makefile

# If CXXFLAGS doesn't exist in Makefile, add it before the build rule
if ! grep -q "CXXFLAGS" Makefile; then
    sed -i '1i CXXFLAGS = -ftemplate-depth=2048' Makefile
fi

# Build the project using make
make clean
make