#!/bin/bash

# Stage 3: Deduplication Pipeline
# This stage removes duplicate IP addresses from the filtered data
# to ensure accurate unique visitor counts

INPUT_FILE="/opt/analytics/data/output/stage2_filtered.txt"
OUTPUT_FILE="/opt/analytics/data/output/stage3_unique.txt"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Deduplicate IP addresses
# Using uniq to remove consecutive duplicate lines
# This should give us unique visitor IPs
cat "$INPUT_FILE" | uniq > "$OUTPUT_FILE"

# Report the count of unique IPs processed
UNIQUE_COUNT=$(wc -l < "$OUTPUT_FILE")
echo "Stage 3 complete: $UNIQUE_COUNT unique IPs identified"