#!/bin/bash

# Production Log Monitor Script
# Processes server logs through multi-stage pipeline to detect errors
# Author: DevOps Team
# Version: 1.0

LOG_DIR="/var/logs/test"
LOG_FILE="${LOG_DIR}/server.log"

echo "=== Server Log Monitor ==="
echo "Processing logs from: ${LOG_FILE}"
echo ""

# Check if log file exists
if [ ! -f "${LOG_FILE}" ]; then
    echo "ERROR: Log file not found: ${LOG_FILE}"
    exit 1
fi

echo "Stage 1: Extracting error entries..."
echo "Stage 2: Filtering by severity level (>= 3)..."
echo "Stage 3: Aggregating statistics..."
echo "Stage 4: Formatting report..."
echo ""

# Multi-stage pipeline to process logs
# Stage 1: Extract all ERROR entries from the log file
# Stage 2: Filter entries where severity level (field 3) is >= 3
# Stage 3: Sort entries and count occurrences
# Stage 4: Format the final report with error type and count
grep 'ERROR' "${LOG_FILE}" | awk '$3 >= 3 {print $2}' | sort | uniq -c | awk '{print $2": "$1" occurrences"}'

# BUG: Only checking the exit status of the last command (awk)
# If grep, first awk, sort, or uniq fail, we won't detect it!
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Report Complete ==="
    echo "Log processing finished successfully"
    exit 0
else
    echo ""
    echo "ERROR: Report generation failed"
    exit 1
fi

# Additional pipeline for summary statistics
echo ""
echo "=== Summary Statistics ==="
grep 'ERROR' "${LOG_FILE}" | wc -l | awk '{print "Total errors found: "$1}'

echo ""
echo "Critical errors (severity >= 3):"
grep 'ERROR' "${LOG_FILE}" | awk '$3 >= 3' | wc -l | awk '{print "Count: "$1}'

# Again, only the last command status is checked
if [ $? -eq 0 ]; then
    echo ""
    echo "Monitoring complete"
    exit 0
else
    echo "Failed to generate summary"
    exit 1
fi