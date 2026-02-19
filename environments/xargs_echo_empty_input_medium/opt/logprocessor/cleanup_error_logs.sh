#!/bin/bash

# Compresses error logs older than 45 days
# This script correctly handles empty input using xargs -r flag

# Set strict error handling
set -o pipefail

# Store the find results in a variable to check if any files exist
FOUND_FILES=$(find /var/app/logs/ -name "error-*.log" -mtime +45)

# Only proceed if files were found
if [ -n "$FOUND_FILES" ]; then
    # Count files for logging
    FILE_COUNT=$(echo "$FOUND_FILES" | wc -l)
    
    # Compress the files using xargs -r (which prevents execution on empty input)
    echo "$FOUND_FILES" | xargs -r gzip -9
    
    # Log the processing action only when files were actually processed
    echo "Processed $FILE_COUNT error log files at $(date)" >> /var/app/logs/archive.log
fi