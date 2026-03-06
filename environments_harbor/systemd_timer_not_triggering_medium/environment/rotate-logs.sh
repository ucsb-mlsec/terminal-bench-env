#!/bin/bash
#
# Log Rotation Script
# Compresses and archives old application logs
# This script runs every 6 hours via cron to manage log files
#

# Define directories
LOG_DIR="/var/log/myapp"
ARCHIVE_DIR="/var/log/myapp/archive"

# Create archive directory if it doesn't exist
mkdir -p "$ARCHIVE_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create archive directory: $ARCHIVE_DIR" >&2
    exit 1
fi

# Generate timestamp for archive filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Counter for processed files
PROCESSED=0

# Find and compress log files older than 6 hours
echo "Starting log rotation at $(date)"

find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -mmin +360 | while read -r logfile; do
    if [ -f "$logfile" ]; then
        BASENAME=$(basename "$logfile" .log)
        ARCHIVE_NAME="${BASENAME}_${TIMESTAMP}.log.gz"
        
        # Compress the log file
        gzip -c "$logfile" > "$ARCHIVE_DIR/$ARCHIVE_NAME"
        
        if [ $? -eq 0 ]; then
            # Remove original file after successful compression
            rm -f "$logfile"
            echo "Archived: $logfile -> $ARCHIVE_DIR/$ARCHIVE_NAME"
            PROCESSED=$((PROCESSED + 1))
        else
            echo "ERROR: Failed to compress $logfile" >&2
        fi
    fi
done

echo "Log rotation completed. Files processed: $PROCESSED"
exit 0