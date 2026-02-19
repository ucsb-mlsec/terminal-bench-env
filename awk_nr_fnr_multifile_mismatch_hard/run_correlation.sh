#!/bin/bash

set -euo pipefail

# Distributed System Log Correlation Runner
# Processes multiple service logs to detect anomalous request patterns

LOG_DIR="/var/log/services"
SCRIPT_DIR="/home/user/scripts"
OUTPUT_DIR="/home/user/output"
AWK_SCRIPT="$SCRIPT_DIR/correlate_logs.awk"
OUTPUT_FILE="$OUTPUT_DIR/anomalies.txt"

# Log files
AUTH_LOG="$LOG_DIR/auth.log"
API_LOG="$LOG_DIR/api.log"
DATABASE_LOG="$LOG_DIR/database.log"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Verify log directory exists
if [[ ! -d "$LOG_DIR" ]]; then
    echo "Error: Log directory $LOG_DIR does not exist" >&2
    exit 1
fi

# Verify AWK script exists
if [[ ! -f "$AWK_SCRIPT" ]]; then
    echo "Error: AWK script $AWK_SCRIPT not found" >&2
    exit 1
fi

# Verify all log files exist
for log_file in "$AUTH_LOG" "$API_LOG" "$DATABASE_LOG"; do
    if [[ ! -f "$log_file" ]]; then
        echo "Error: Log file $log_file not found" >&2
        exit 1
    fi
    if [[ ! -r "$log_file" ]]; then
        echo "Error: Log file $log_file is not readable" >&2
        exit 1
    fi
done

# Run the correlation analysis
echo "Starting log correlation analysis..." >&2
echo "Processing logs from: $LOG_DIR" >&2
echo "Using AWK script: $AWK_SCRIPT" >&2

if awk -f "$AWK_SCRIPT" \
    "$AUTH_LOG" \
    "$API_LOG" \
    "$DATABASE_LOG" > "$OUTPUT_FILE"; then
    
    echo "Correlation analysis complete" >&2
    echo "Results saved to: $OUTPUT_FILE" >&2
    
    # Display summary
    if [[ -f "$OUTPUT_FILE" ]]; then
        anomaly_count=$(wc -l < "$OUTPUT_FILE")
        echo "Total anomalies detected: $anomaly_count" >&2
    fi
    
    exit 0
else
    echo "Error: Correlation analysis failed" >&2
    exit 1
fi