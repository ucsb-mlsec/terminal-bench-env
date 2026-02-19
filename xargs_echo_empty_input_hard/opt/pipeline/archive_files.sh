#!/bin/bash

# archive_files.sh
# Archives old transaction files to reduce storage costs
# This script runs nightly to compress and move aged transaction data

# Archive transactions older than 30 days
echo "Archiving old transaction files"

# Find and compress old transaction files
# Files older than 30 days are moved to compressed archives
find /var/data/transactions/ -name "*.txn" -mtime +30 | xargs tar -czf /var/archive/transactions_$(date +%Y%m%d).tar.gz

# Move completed transactions to archive directory
# This keeps the active processing directory clean
find /var/data/transactions/completed/ -type f | xargs -I {} mv {} /var/archive/

# Log the archive operation
echo "Archive operation completed at $(date)"

# Update archive index for quick retrieval
echo "Updating archive index..."

# Clean up temporary files older than 7 days
find /tmp/txn_* -mtime +7 -type f 2>/dev/null

echo "Archive process finished successfully"