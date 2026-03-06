#!/bin/bash

# Data Processing Pipeline Script
# Processes CSV files through multiple transformation stages

INPUT_DIR="/opt/dataproc/input"
OUTPUT_DIR="/opt/dataproc/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# PIPELINE 1
# Extract and sort customer data by revenue
cat "$INPUT_DIR/customers.csv" | grep -v "^#" | awk -F',' '{print $1","$2","$5}' | sort -t',' -k3 -nr | head -100 > "$OUTPUT_DIR/top_customers.csv"

# Check if we have any input files
if [ ! -f "$INPUT_DIR/customers.csv" ]; then
    echo "Warning: customers.csv not found"
fi

# PIPELINE 2
# Process transaction data and calculate totals
cat "$INPUT_DIR/transactions.csv" | sed 's/\r$//' | awk -F',' '$4 > 100 {print $0}' | cut -d',' -f1,2,4,5 | sort -t',' -k1 > "$OUTPUT_DIR/high_value_transactions.csv"

# Intermediate processing for report generation
TEMP_FILE=$(mktemp)
if [ -f "$INPUT_DIR/transactions.csv" ]; then
    cp "$INPUT_DIR/transactions.csv" "$TEMP_FILE"
fi

# PIPELINE 3
# Generate summary report with aggregated metrics
cat "$INPUT_DIR/sales.csv" | grep -E "^[0-9]" | awk -F',' '{sum[$2]+=$3; count[$2]++} END {for(cat in sum) print cat","sum[cat]","count[cat]}' | sort -t',' -k2 -nr | head -50 > "$OUTPUT_DIR/sales_summary.csv"

# Cleanup temporary files
rm -f "$TEMP_FILE"

# Additional processing metadata
PROCESSED_COUNT=$(ls -1 "$INPUT_DIR"/*.csv 2>/dev/null | wc -l)

# Generate processing timestamp
date +%Y-%m-%d_%H:%M:%S > "$OUTPUT_DIR/.last_run"

# Final validation check
if [ -f "$OUTPUT_DIR/top_customers.csv" ]; then
    LINES=$(wc -l < "$OUTPUT_DIR/top_customers.csv")
fi

if [ -f "$OUTPUT_DIR/high_value_transactions.csv" ]; then
    TRANS_LINES=$(wc -l < "$OUTPUT_DIR/high_value_transactions.csv")
fi

if [ -f "$OUTPUT_DIR/sales_summary.csv" ]; then
    SUMMARY_LINES=$(wc -l < "$OUTPUT_DIR/sales_summary.csv")
fi

# Script completes successfully
exit 0