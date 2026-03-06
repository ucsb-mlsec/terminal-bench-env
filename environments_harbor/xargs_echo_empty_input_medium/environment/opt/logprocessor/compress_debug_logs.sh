#!/bin/bash
# Compress debug logs older than 7 days

# Find and compress old debug log files
find /var/app/logs/ -name "debug-*.log" -mtime +7 | xargs bzip2

# Log the processing action
echo "Processed 0 files at $(date)" >> /var/app/logs/archive.log