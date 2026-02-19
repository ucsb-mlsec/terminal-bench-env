#!/bin/bash

# Deduplication script for distributed monitoring system
# Purpose: Merge and deduplicate log files from multiple servers
# Author: DevOps Team
# Date: 2024-01-15

# Set the log directory
LOG_DIR="/var/logs/app"
OUTPUT_FILE="/tmp/deduplicated.log"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory $LOG_DIR does not exist"
    exit 1
fi

# Concatenate all log files and remove duplicates
# Using sort and uniq to eliminate duplicate lines
cat $LOG_DIR/*.log | sort | uniq > $OUTPUT_FILE

# Sort the output by timestamp for better readability
sort -o $OUTPUT_FILE $OUTPUT_FILE

echo "Deduplication complete. Output written to $OUTPUT_FILE"
echo "Total unique events: $(wc -l < $OUTPUT_FILE)"