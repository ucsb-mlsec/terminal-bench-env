#!/bin/bash

# Process Count Monitor
# Checks if process count is within acceptable range

DATA_FILE="/opt/monitoring/data/processes.txt"
MIN_THRESHOLD=10
MAX_THRESHOLD=200

# Read process count from data file
if [ -f "$DATA_FILE" ]; then
    process_count=$(cat "$DATA_FILE")
else
    echo "ERROR: Data file not found"
    exit 1
fi

# Check if value is empty
if [ -z "$process_count" ]; then
    echo "ERROR: No process count data available"
    exit 1
fi

# Check for minimum threshold - WRONG: using string comparison operator
if [ "$process_count" < "$MIN_THRESHOLD" ]; then
    echo "ALERT: Too few processes running: $process_count"
fi

# Check for maximum threshold - WRONG: using quotes with -gt
if [ "$process_count" -gt "$MAX_THRESHOLD" ]; then
    echo "ALERT: Too many processes running: $process_count"
fi

# Check if within valid range - WRONG: using deprecated -a operator
if [ "$process_count" -ge "$MIN_THRESHOLD" -a "$process_count" -le "$MAX_THRESHOLD" ]; then
    echo "OK: Process count is normal: $process_count"
fi

exit 0