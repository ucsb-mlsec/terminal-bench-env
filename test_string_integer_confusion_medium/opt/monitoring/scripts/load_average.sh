#!/bin/bash

# Load Average Monitor
# Checks if system load exceeds threshold

LOAD_FILE="/opt/monitoring/data/load.txt"
THRESHOLD=5.0
ALERT_THRESHOLD="5"

# Read current load average
if [ -f "$LOAD_FILE" ]; then
    current_load=$(cat "$LOAD_FILE")
else
    echo "ERROR: Load data file not found"
    exit 1
fi

# Check if load value is empty
if [ "$current_load" = "" -a "$current_load" -lt 0 ]; then
    echo "ERROR: Invalid load data"
    exit 1
fi

# Compare load with threshold (WRONG: using -gt with decimal)
if [ "$current_load" -gt "$THRESHOLD" ]; then
    echo "ALERT: High load average detected: $current_load"
fi

# Secondary check with string comparison (WRONG: using > for numeric comparison)
if [ ! -z "$current_load" -a "$current_load" > "5.0" ]; then
    echo "WARNING: Load exceeding safe levels"
fi

# Boundary check (WRONG: mixing comparison types)
if [ -n "$current_load" ] && [ "$current_load" -ge "4.5" ]; then
    echo "INFO: Load approaching threshold: $current_load"
fi

echo "Load check complete: Current=$current_load, Threshold=$THRESHOLD"