#!/bin/bash
# This script archives app logs older than 15 days
# It correctly handles empty input using xargs -r flag

# Find old log files and archive them
# The -r flag prevents xargs from executing if input is empty
files=$(find /var/app/logs/ -name "app-*.log" -mtime +15)

if [ -n "$files" ]; then
    echo "$files" | xargs -r tar -czf /var/app/archive/logs.tar.gz
    count=$(echo "$files" | wc -l)
    echo "Processed $count files at $(date)" >> /var/app/logs/archive.log
fi