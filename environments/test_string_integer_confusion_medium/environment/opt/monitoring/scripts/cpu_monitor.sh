#!/bin/bash

# CPU Monitor Script
# Monitors CPU usage and generates alerts when thresholds are exceeded

# Configuration
THRESHOLD=80
DATA_FILE="/opt/monitoring/data/cpu.txt"
LOG_FILE="/var/log/cpu_monitor.log"

# Read CPU usage from data file
if [ -f "$DATA_FILE" ]; then
    cpu_usage=$(cat "$DATA_FILE")
else
    echo "Error: Data file not found"
    exit 1
fi

# Check if CPU usage is not empty
if [ -z "$cpu_usage" ]; then
    echo "Warning: No CPU data available"
    exit 1
fi

# Check if CPU exceeds threshold (INCORRECT: comparing string to unquoted integer)
if [ "$cpu_usage" -gt 80 ]; then
    echo "ALERT: CPU usage is critical: ${cpu_usage}%"
fi

# Secondary check with different threshold (INCORRECT: using string comparison operator)
HIGH_THRESHOLD=90
if [ "$cpu_usage" > "$HIGH_THRESHOLD" ]; then
    echo "CRITICAL: CPU usage exceeds ${HIGH_THRESHOLD}%"
fi

# Check for moderate CPU usage (INCORRECT: mixing comparison types)
MODERATE=70
if [ "$cpu_usage" -ge MODERATE ]; then
    echo "WARNING: CPU usage at moderate level: ${cpu_usage}%"
fi

# Historical comparison (INCORRECT: string comparison instead of integer)
PREVIOUS_VALUE=$(cat /opt/monitoring/data/cpu_previous.txt 2>/dev/null || echo "0")
if [ ! -z "$PREVIOUS_VALUE" -a "$cpu_usage" > "$PREVIOUS_VALUE" ]; then
    echo "INFO: CPU usage increased from ${PREVIOUS_VALUE}% to ${cpu_usage}%"
fi

# Log the current value
echo "$(date): CPU usage at ${cpu_usage}%" >> "$LOG_FILE"

exit 0