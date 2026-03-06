#!/bin/bash

# Simple script to process log files and add timestamps
# Just finds all logs and adds a footer - should be straightforward

LOG_DIR="/var/log_archive"

# Find all log files and process them
# Using -exec to append timestamp to each file
find "$LOG_DIR" -type f -name "*.log" -exec sh -c 'echo "Processed: $(date +"%Y-%m-%d %H:%M:%S")" >> "$1"' _ {} \;

# Done! This should work fine for all our log files
echo "Log processing complete"