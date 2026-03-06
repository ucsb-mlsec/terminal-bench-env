#!/bin/bash

# Stage 1: Extract IP addresses from log files
# This stage reads all .log files from the logs directory and extracts
# IP addresses from the pipe-delimited format: timestamp|ip_address|user_agent|request_path

# Input directory containing raw log files
LOG_DIR="/opt/analytics/data/logs"

# Output file for extracted IP addresses
OUTPUT_FILE="/opt/analytics/data/output/stage1_ips.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Process all .log files in the logs directory
for log_file in "$LOG_DIR"/*.log; do
    # Check if log file exists (handles case where no .log files are found)
    if [ -f "$log_file" ]; then
        # Extract the second field (IP address) from pipe-delimited format
        # and append to the output file
        cut -d'|' -f2 "$log_file" >> "$OUTPUT_FILE"
    fi
done

echo "Stage 1 complete: IP addresses extracted to $OUTPUT_FILE"