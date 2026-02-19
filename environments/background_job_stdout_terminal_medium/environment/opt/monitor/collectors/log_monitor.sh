#!/bin/bash

# Log Monitor - Scans system logs for errors and warnings
# This script continuously monitors system logs and reports error counts

while true; do
    # Print timestamp
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
    
    # Print monitoring status
    echo "Log Monitor: Scanning for errors..."
    
    # Simulate error counting with random-ish values
    ERROR_COUNT=$((RANDOM % 6))
    WARNING_COUNT=$((RANDOM % 10 + 2))
    
    echo "Found $ERROR_COUNT errors in system logs"
    echo "Found $WARNING_COUNT warnings in system logs"
    echo "---"
    
    # Wait 5 seconds before next check
    sleep 5
done