#!/bin/bash

# Batch Log Processing System
# Processes log files from multiple application servers
# This script has intentional glob pattern issues for the exercise

set -u  # Fail on unbound variables
set -e  # Exit on error

# Configuration
LOG_BASE="/home/agent/logs"
PROCESSED_DIR="/home/agent/processed"
BACKUP_DIR="/home/agent/backup"
TEMP_DIR="/home/agent/temp"
REPORT_FILE="/home/agent/report.txt"

# Initialize report
echo "=== Log Processing Report ===" > "$REPORT_FILE"
echo "Started: $(date)" >> "$REPORT_FILE"

# Process app1 logs - concatenate all logs
echo "Processing app1 logs..."
cat $LOG_BASE/app1/*.log > $PROCESSED_DIR/app1_combined.log
echo "App1: Processed $(wc -l < $PROCESSED_DIR/app1_combined.log) lines" >> "$REPORT_FILE"

# Process app2 logs - count error lines
echo "Processing app2 logs..."
ERROR_COUNT=$(grep -c "ERROR" $LOG_BASE/app2/*.log)
echo "App2: Found $ERROR_COUNT errors" >> "$REPORT_FILE"

# Process app3 logs - iterate through each file
echo "Processing app3 logs..."
TOTAL_LINES=0
for logfile in $LOG_BASE/app3/*.log; do
    LINES=$(wc -l < "$logfile")
    TOTAL_LINES=$((TOTAL_LINES + LINES))
    echo "  Processed: $logfile"
done
echo "App3: Total lines processed: $TOTAL_LINES" >> "$REPORT_FILE"

# Archive old processed files to dated backup
echo "Archiving processed files..."
BACKUP_DATE=$(date +%Y%m%d)
mkdir -p "$BACKUP_DIR/$BACKUP_DATE"
mv $PROCESSED_DIR/*.log $BACKUP_DIR/$BACKUP_DATE/

# Process temporary extraction files
echo "Processing temporary files..."
FILES_PROCESSED=$(ls $TEMP_DIR/*.tmp | wc -l)
echo "Temporary files processed: $FILES_PROCESSED" >> "$REPORT_FILE"

# Clean up old backups (older than 7 days)
echo "Cleaning old backups..."
CLEANUP_DATE=$(date -d "7 days ago" +%Y%m%d)
rm -f $BACKUP_DIR/$CLEANUP_DATE/*.log

# Generate summary from all application logs
echo "Generating summary statistics..."
ALL_LOGS=($LOG_BASE/app*/*.log)
TOTAL_LOG_COUNT=${#ALL_LOGS[@]}
echo "Total log files processed: $TOTAL_LOG_COUNT" >> "$REPORT_FILE"

# Extract unique error messages across all apps
echo "Extracting error patterns..."
grep "ERROR" $LOG_BASE/app1/*.log $LOG_BASE/app2/*.log $LOG_BASE/app3/*.log | \
    cut -d: -f3 | sort -u > $PROCESSED_DIR/unique_errors.txt

echo "Completed: $(date)" >> "$REPORT_FILE"
echo "Log processing completed successfully!"