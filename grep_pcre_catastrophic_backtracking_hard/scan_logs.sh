#!/bin/bash

# Configuration
PATTERN_FILE="/opt/analyzer/error_patterns.txt"
LOG_DIR="/var/logs/application"
LOG_FILES=("service_2024_01.log" "service_2024_02.log" "service_2024_03.log")

# Initialize counters
total_matches=0

echo "Starting log analysis..."
echo "Pattern file: $PATTERN_FILE"
echo "Log directory: $LOG_DIR"
echo ""

# Check if pattern file exists
if [ ! -f "$PATTERN_FILE" ]; then
    echo "Error: Pattern file not found: $PATTERN_FILE"
    exit 1
fi

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory not found: $LOG_DIR"
    exit 1
fi

# Read patterns and search logs
while IFS= read -r pattern || [ -n "$pattern" ]; do
    # Skip empty lines and comments
    [[ -z "$pattern" || "$pattern" =~ ^# ]] && continue
    
    echo "Checking pattern: $pattern"
    
    for logfile in "${LOG_FILES[@]}"; do
        log_path="$LOG_DIR/$logfile"
        if [ -f "$log_path" ]; then
            # Use grep with PCRE mode
            matches=$(grep -P "$pattern" "$log_path" 2>/dev/null | wc -l)
            if [ $matches -gt 0 ]; then
                echo "  Found $matches matches in $logfile"
                total_matches=$((total_matches + matches))
            fi
        else
            echo "  Warning: Log file not found: $log_path"
        fi
    done
done < "$PATTERN_FILE"

echo ""
echo "Analysis complete. Total matches: $total_matches"