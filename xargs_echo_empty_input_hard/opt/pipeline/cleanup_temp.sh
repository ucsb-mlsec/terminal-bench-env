#!/bin/bash

# cleanup_temp.sh
# This script should always run to ensure cleanup
# Cleans up temporary transaction files from the processing pipeline

echo "Starting cleanup of temporary files"

# Remove temporary files older than 1 day
# Find all .tmp files that are more than 1 day old and remove them
find /var/data/transactions/temp/ -name "*.tmp" -mtime +1 | xargs rm -f

# Check for any processing lock files
find /var/data/transactions/temp/ -name "*.lock" -mtime +0 | xargs rm -f

# Remove empty directories from temp area
# This already has -r flag to avoid running on empty input
find /var/data/transactions/temp/ -type d -empty | xargs -r rmdir

# Log cleanup operation
echo "Cleanup operation executed at $(date)" >> /var/log/pipeline/cleanup.log

echo "Cleanup complete"