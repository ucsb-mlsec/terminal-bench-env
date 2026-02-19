#!/bin/bash

# cleanup.sh - Cleans up old processed log files
# This script removes log files older than specified days

# Directory containing processed logs
LOG_DIR="/opt/log_processor/processed"
DEFAULT_DAYS=7

# Get cleanup age from argument or use default
if [ $# -gt 0 ]; then
    CLEANUP_DAYS=$1
else
    CLEANUP_DAYS=$DEFAULT_DAYS
fi

# Get optional file pattern from second argument
if [ $# -gt 1 ]; then
    FILE_PATTERN=$2
else
    FILE_PATTERN="*.log"
fi

echo "Starting cleanup of logs older than $CLEANUP_DAYS days..."

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# VULNERABLE: Using eval with user-controllable input
# Build cleanup command dynamically using eval
CLEANUP_COMMAND="find $LOG_DIR -name '$FILE_PATTERN' -type f -mtime +$CLEANUP_DAYS"
eval "$CLEANUP_COMMAND" > /tmp/files_to_clean.txt

# Count files to be removed
FILE_COUNT=$(wc -l < /tmp/files_to_clean.txt)
echo "Found $FILE_COUNT files to remove"

# VULNERABLE: Using eval to execute removal command
# This allows command injection through FILE_PATTERN or CLEANUP_DAYS
while IFS= read -r file; do
    if [ -f "$file" ]; then
        REMOVE_CMD="rm -f $file"
        eval "$REMOVE_CMD"
        echo "Removed: $file"
    fi
done < /tmp/files_to_clean.txt

# Clean up temporary file
rm -f /tmp/files_to_clean.txt

echo "Cleanup completed: $FILE_COUNT files removed"