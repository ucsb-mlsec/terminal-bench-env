#!/bin/bash
set -e

echo "Compiling C++ template library..."
g++ -std=c++17 -I. -Wall -Wextra -o test_containers test_containers.cpp

echo "Build successful!"
./test_containers
exit_code=$?

echo "Test execution completed with exit code: $exit_code"
exit $exit_code