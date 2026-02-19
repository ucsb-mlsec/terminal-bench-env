#!/bin/bash
# /opt/logprocessor/process_logs.sh
# Log processing script with BUGGY directory creation (race condition)

set -e

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_log_file> <output_date_path>" >&2
    echo "Example: $0 /var/logs/incoming/app.log 2024/01/15" >&2
    exit 1
fi

INPUT_LOG="$1"
DATE_PATH="$2"
OUTPUT_DIR="/var/logs/processed/${DATE_PATH}"

# Verify input file exists
if [ ! -f "$INPUT_LOG" ]; then
    echo "Error: Input log file not found: $INPUT_LOG" >&2
    exit 1
fi

# BUGGY DIRECTORY CREATION - RACE CONDITION HERE!
# This is the classic TOCTOU (Time-Of-Check-Time-Of-Use) bug
# Problem: Multiple workers check if directory exists, all see it doesn't exist,
# then all try to create it, causing "File exists" errors

# Step 1: Check if directory exists (TIME OF CHECK)
if [ ! -d "$OUTPUT_DIR" ]; then
    # Step 2: Create directory (TIME OF USE)
    # RACE CONDITION: Another process might create this directory between check and create!
    # This will fail with "File exists" error when multiple processes race
    mkdir -p "$OUTPUT_DIR"
fi

# Extract filename for output
BASENAME=$(basename "$INPUT_LOG")
OUTPUT_FILE="${OUTPUT_DIR}/processed_${BASENAME}"

# Process the log file (simple processing: add timestamp and copy)
{
    echo "# Processed at $(date)"
    echo "# Source: $INPUT_LOG"
    echo "# Worker PID: $$"
    echo "---"
    cat "$INPUT_LOG"
} > "$OUTPUT_FILE"

echo "Successfully processed $INPUT_LOG -> $OUTPUT_FILE"
exit 0