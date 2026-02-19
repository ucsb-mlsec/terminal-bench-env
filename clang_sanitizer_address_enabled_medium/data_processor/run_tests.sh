#!/bin/bash

echo "Running data processor tests..."
echo "================================"
echo ""

# Run tests and capture output, but don't exit on error
set +e

echo "Test 1: Running with no arguments"
./data_processor 2>&1
echo ""

echo "Test 2: Basic processing with small input"
./data_processor "hello" 2>&1
echo ""

echo "Test 3: Processing with numeric input"
./data_processor "12345" 2>&1
echo ""

echo "Test 4: Processing with larger input"
./data_processor "This is a longer test string with multiple words" 2>&1
echo ""

echo "Test 5: Edge case - empty string"
./data_processor "" 2>&1
echo ""

echo "Test 6: Edge case - special characters"
./data_processor "test@#$%^&*()" 2>&1
echo ""

echo "Test 7: Multiple arguments"
./data_processor "arg1" "arg2" "arg3" 2>&1
echo ""

echo "================================"
echo "Test suite completed"
echo ""