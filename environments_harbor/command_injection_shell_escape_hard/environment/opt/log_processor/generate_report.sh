#!/bin/bash
# generate_report.sh - Generates summary reports from processed log files
# This script aggregates log data and creates formatted reports

REPORT_DIR="/var/log/reports"
LOG_DIR="/var/log/processed"

# Function to display usage
usage() {
    echo "Usage: $0 <log_file> <report_name>"
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

LOG_FILE="$1"
REPORT_NAME="$2"

# Verify log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi

# Create report directory if needed
mkdir -p "$REPORT_DIR"

# Generate report header with timestamp
REPORT_FILE="$REPORT_DIR/${REPORT_NAME}_report.txt"
echo "=== Log Analysis Report ===" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "Source: $LOG_FILE" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# VULNERABLE: Command substitution with user-controlled report name
HEADER_TEXT=`echo "Report for: $REPORT_NAME"`
echo "$HEADER_TEXT" >> "$REPORT_FILE"

# Count total log entries
TOTAL_LINES=$(wc -l < "$LOG_FILE")
echo "Total entries: $TOTAL_LINES" >> "$REPORT_FILE"

# VULNERABLE: Using eval with user data from log file
# Read the first line to get log type/category
LOG_CATEGORY=$(head -n 1 "$LOG_FILE" | cut -d: -f1)
CATEGORY_DISPLAY=$(eval echo "Category: $LOG_CATEGORY")
echo "$CATEGORY_DISPLAY" >> "$REPORT_FILE"

# Count error and warning levels
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" || echo 0)
WARN_COUNT=$(grep -c "WARN" "$LOG_FILE" || echo 0)
INFO_COUNT=$(grep -c "INFO" "$LOG_FILE" || echo 0)

echo "" >> "$REPORT_FILE"
echo "Statistics:" >> "$REPORT_FILE"
echo "  Errors: $ERROR_COUNT" >> "$REPORT_FILE"
echo "  Warnings: $WARN_COUNT" >> "$REPORT_FILE"
echo "  Info: $INFO_COUNT" >> "$REPORT_FILE"

echo "" >> "$REPORT_FILE"
echo "Report saved to: $REPORT_FILE"