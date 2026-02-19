#!/bin/bash
# Remove temporary log files older than 1 day

find /var/app/logs/temp/ -name "*.tmp" -mtime +1 | xargs rm -f

# Log processing completion
echo "Processed 0 files at $(date +"%Y-%m-%d %H:%M:%S")" >> /var/app/logs/archive.log