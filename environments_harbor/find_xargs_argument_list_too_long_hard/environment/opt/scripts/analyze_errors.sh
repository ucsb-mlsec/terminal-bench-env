#!/bin/bash
# analyze_errors.sh - Analyzes error patterns across all application logs
# This script searches for ERROR patterns in log files and reports statistics

LOG_DIR="/var/log/application"
TEMP_FILE="/tmp/error_analysis_$$.tmp"

echo "=== Application Error Analysis Report ==="
echo "Analyzing logs in: $LOG_DIR"
echo "Timestamp: $(date)"
echo ""

# Problematic approach: passing all files as arguments to grep
# This will fail with "Argument list too long" when there are many files
echo "Searching for ERROR patterns..."
grep -h "ERROR" $LOG_DIR/*.log > $TEMP_FILE 2>/dev/null

# Count total errors
TOTAL_ERRORS=$(wc -l < $TEMP_FILE)
echo "Total ERROR entries found: $TOTAL_ERRORS"

# Count unique error types (simplified)
echo "Top 10 most common errors:"
cut -d':' -f3- $TEMP_FILE | sort | uniq -c | sort -rn | head -10

# Clean up
rm -f $TEMP_FILE

echo ""
echo "Analysis complete."