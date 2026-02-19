#!/bin/bash

# Data Pipeline Join Script
# Purpose: Join customer, transaction, and product data for daily sales reporting
# This script contains errors in join field specifications that need to be fixed

INPUT_DIR="/opt/data_pipeline/input"
OUTPUT_DIR="/opt/data_pipeline/output"
LOG_FILE="/opt/data_pipeline/logs/pipeline.log"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting data pipeline at $(date)" > "$LOG_FILE"
echo "Input directory: $INPUT_DIR" >> "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" >> "$LOG_FILE"

# Check if input files exist
if [[ ! -f "$INPUT_DIR/customers.csv" ]]; then
    echo "ERROR: customers.csv not found" >> "$LOG_FILE"
    exit 1
fi

if [[ ! -f "$INPUT_DIR/transactions.csv" ]]; then
    echo "ERROR: transactions.csv not found" >> "$LOG_FILE"
    exit 1
fi

if [[ ! -f "$INPUT_DIR/products.csv" ]]; then
    echo "ERROR: products.csv not found" >> "$LOG_FILE"
    exit 1
fi

echo "All input files found" >> "$LOG_FILE"

# Step 1: Sort customers by customer_id (field 2 - WRONG, should be field 1)
echo "Sorting customers.csv..." >> "$LOG_FILE"
sort -t',' -k2 "$INPUT_DIR/customers.csv" > /tmp/customers_sorted.csv

# Step 2: Sort transactions by customer_id (field 2 - WRONG, should be field 1)
echo "Sorting transactions.csv..." >> "$LOG_FILE"
sort -t',' -k2 "$INPUT_DIR/transactions.csv" > /tmp/transactions_sorted.csv

# Step 3: Join customers with transactions
# This should join on customer_id which is field 1 in both files
# But we're using WRONG field numbers: -1 2 (should be 1) and -2 3 (should be 1)
echo "Joining customers with transactions..." >> "$LOG_FILE"
join -t',' -1 2 -2 3 /tmp/customers_sorted.csv /tmp/transactions_sorted.csv > /tmp/cust_trans.csv 2>> "$LOG_FILE"

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to join customers with transactions" >> "$LOG_FILE"
fi

# Step 4: Sort the intermediate result for joining with products
# Sorting by product_id field (field 4 - WRONG, should be field 3 after first join)
echo "Sorting intermediate result..." >> "$LOG_FILE"
sort -t',' -k4 /tmp/cust_trans.csv > /tmp/cust_trans_sorted.csv

# Step 5: Sort products by product_id (field 1 - WRONG, should be different)
echo "Sorting products.csv..." >> "$LOG_FILE"
sort -t',' -k1 "$INPUT_DIR/products.csv" > /tmp/products_sorted.csv

# Step 6: Join the combined customer-transaction data with products
# This should join on product_id
# But we're using WRONG field numbers: -1 4 (should be 3) and -2 2 (should be 1)
echo "Joining with products..." >> "$LOG_FILE"
join -t',' -1 4 -2 2 /tmp/cust_trans_sorted.csv /tmp/products_sorted.csv > "$OUTPUT_DIR/final_report.csv" 2>> "$LOG_FILE"

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to join with products" >> "$LOG_FILE"
fi

# Count output records
OUTPUT_LINES=$(wc -l < "$OUTPUT_DIR/final_report.csv" 2>/dev/null || echo 0)
echo "Pipeline completed at $(date)" >> "$LOG_FILE"
echo "Output records: $OUTPUT_LINES" >> "$LOG_FILE"

if [[ $OUTPUT_LINES -eq 0 ]]; then
    echo "WARNING: No records in output file - join operations likely failed" >> "$LOG_FILE"
    exit 1
fi

echo "Final report saved to: $OUTPUT_DIR/final_report.csv" >> "$LOG_FILE"