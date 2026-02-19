#!/bin/bash
# Archives application logs older than 30 days by compressing them

# Find and compress old log files
find /var/app/logs/ -name "*.log" -mtime +30 | xargs gzip

# Log the processing action
echo "Processed logs at $(date)" >> /var/app/logs/archive.log