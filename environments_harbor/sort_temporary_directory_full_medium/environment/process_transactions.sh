#!/bin/bash

# Data processing pipeline for transaction sorting
# This script processes transaction records from CSV files

set -e

# Define input and output paths
INPUT_FILE="/data/input/transactions.csv"
OUTPUT_FILE="/data/output/transactions_sorted.csv"

echo "Starting transaction processing pipeline..."
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# Sorting transaction records
# Using comma delimiter and sorting by timestamp (column 2)
sort -t',' -k2 "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Transaction processing completed successfully"