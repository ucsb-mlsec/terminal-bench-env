#!/bin/bash

# Broken Log Processing and Monitoring Script
# This script processes application logs but has critical variable scope issues

LOG_DIR="/var/logs/app"

# Initialize counters and data structures
total_errors=0
critical_count=0
declare -A error_frequencies

# Function to process a single log file
process_log_file() {
    local logfile=$1
    
    # Extract error lines and process them
    # BUG: This pipeline creates a subshell, variables updated inside won't persist
    cat "$logfile" | grep "ERROR" | while read -r line; do
        # Increment total error counter
        total_errors=$((total_errors + 1))
        
        # Parse the log line
        # Format: [TIMESTAMP] [SERVER_ID] [SEVERITY] ERROR_CODE: message
        severity=$(echo "$line" | sed -n 's/.*\[\([0-9]\+\)\].*/\1/p')
        error_code=$(echo "$line" | sed -n 's/.*\] \(E[0-9]\+\):.*/\1/p')
        
        # Check if this is a critical error
        if [ -n "$severity" ] && [ "$severity" -ge 8 ]; then
            critical_count=$((critical_count + 1))
        fi
        
        # Track error code frequencies
        if [ -n "$error_code" ]; then
            if [ -z "${error_frequencies[$error_code]}" ]; then
                error_frequencies[$error_code]=1
            else
                error_frequencies[$error_code]=$((error_frequencies[$error_code] + 1))
            fi
        fi
    done
}

# Main processing loop
# BUG: Another subshell scope issue with find pipeline
find "$LOG_DIR" -name "*.log" -type f | while read -r logfile; do
    process_log_file "$logfile"
done

# Attempt to find the most frequent error code
# BUG: error_frequencies array is empty due to subshell scope issues
top_error=""
max_count=0

for error_code in "${!error_frequencies[@]}"; do
    count=${error_frequencies[$error_code]}
    if [ "$count" -gt "$max_count" ]; then
        max_count=$count
        top_error=$error_code
    fi
done

# Generate JSON output
# BUG: All counters will be 0 or empty because updates were lost in subshells
echo "{"
echo "  \"total_errors\": $total_errors,"
echo "  \"critical_count\": $critical_count,"
if [ -n "$top_error" ]; then
    echo "  \"top_error\": \"$top_error\""
else
    echo "  \"top_error\": \"\""
fi
echo "}"

exit 0