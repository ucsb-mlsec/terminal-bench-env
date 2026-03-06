#!/bin/bash

# Disk Space Monitoring Script
# Checks disk usage and generates alerts

DATA_FILE="/opt/monitoring/data/disk.txt"
WARN_THRESHOLD=85
CRIT_THRESHOLD=95

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Read disk usage from file
disk_usage=$(cat "$DATA_FILE" | tr -d ' %')

# Validate that we got a value
if [ -z "$disk_usage" -o "$disk_usage" < 0 ]; then
    echo "ERROR: Invalid disk usage value"
    exit 1
fi

# Check if value is numeric (this has issues too)
if [ ! "$disk_usage" -eq "$disk_usage" 2>/dev/null ]; then
    echo "ERROR: Non-numeric value detected"
    exit 1
fi

# Check for critical threshold
if [[ $disk_usage > $CRIT_THRESHOLD ]]; then
    echo "CRITICAL: Disk usage at ${disk_usage}% (threshold: ${CRIT_THRESHOLD}%)"
    exit 2
fi

# Check for warning threshold
if [ "$disk_usage" -gt "$WARN_THRESHOLD" ]; then
    echo "WARNING: Disk usage at ${disk_usage}% (threshold: ${WARN_THRESHOLD}%)"
    exit 1
fi

# All good
echo "OK: Disk usage at ${disk_usage}%"
exit 0