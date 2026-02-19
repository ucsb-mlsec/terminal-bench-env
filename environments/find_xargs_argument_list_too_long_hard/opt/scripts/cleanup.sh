#!/bin/bash
# cleanup.sh - Compresses log files older than 30 days
# This script processes log files in /var/log/application directory
# and compresses them using gzip

LOG_DIR="/var/log/application"
DAYS_OLD=30

# Calculate the date 30 days ago
CUTOFF_DATE=$(date -d "${DAYS_OLD} days ago" +%Y%m%d)

echo "Starting cleanup of logs older than ${DAYS_OLD} days..."
echo "Cutoff date: ${CUTOFF_DATE}"

cd "${LOG_DIR}" || exit 1

# This will fail with "Argument list too long" when there are too many files
# because the shell expands *.log into all filenames before passing to the command
OLD_LOGS=$(ls app_*.log 2>/dev/null | while read -r file; do
    # Extract date from filename (app_YYYYMMDD_HHMMSS_<node_id>.log)
    FILE_DATE=$(echo "$file" | sed 's/app_\([0-9]\{8\}\).*/\1/')
    if [ "${FILE_DATE}" -lt "${CUTOFF_DATE}" ]; then
        echo "$file"
    fi
done)

# This is the problematic line - passing all files as arguments at once
gzip ${OLD_LOGS}

echo "Cleanup completed successfully"