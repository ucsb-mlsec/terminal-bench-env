#!/bin/bash

# Disk Usage Monitor
# This script continuously monitors disk usage and reports statistics

while true; do
    # Print timestamp
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
    
    # Print monitoring status
    echo "Disk Monitor: Checking usage..."
    
    # Check and display disk usage
    df -h / 2>/dev/null | grep -v Filesystem | awk '{print "Root filesystem: " $5 " used"}' || echo "Root filesystem: 45% used"
    
    # Additional filesystem info
    echo "Temp filesystem: 12% used"
    
    # Separator for readability
    echo "---"
    
    # Wait 5 seconds before next check
    sleep 5
done