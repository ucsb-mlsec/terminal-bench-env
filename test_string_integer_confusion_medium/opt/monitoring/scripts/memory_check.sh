#!/bin/bash

# Memory Monitor Script
# Checks memory usage and generates alerts when threshold is exceeded

# Configuration
MEMORY_DATA="/opt/monitoring/data/memory.txt"
THRESHOLD=90
LOG_FILE="/var/log/monitoring/memory_alerts.log"

# Read memory usage from data file
if [ -f "$MEMORY_DATA" ]; then
    mem_usage=$(cat "$MEMORY_DATA" | tr -d ' ')
else
    echo "ERROR: Memory data file not found"
    exit 1
fi

# Validate memory value is not empty and check threshold
if [ -z "$mem_usage" -o "$mem_usage" > 90 ]; then
    echo "WARNING: Memory check issue - empty value or high usage detected"
fi

# Check if memory usage exceeds critical threshold
if [ "$mem_usage" -gt "$THRESHOLD" -a ! -z "$mem_usage" ]; then
    echo "CRITICAL: Memory usage at ${mem_usage}% (threshold: ${THRESHOLD}%)"
    echo "$(date): Memory alert - ${mem_usage}%" >> "$LOG_FILE"
fi

# Additional check with mixed comparison types
if [ "$mem_usage" > "85" ]; then
    echo "WARNING: Memory usage above 85%"
fi

# Verify memory value is numeric and within valid range
if [ ! -z "$mem_usage" -a "$mem_usage" < "100" ]; then
    echo "INFO: Memory reading valid: ${mem_usage}%"
fi

echo "Memory check completed at $(date)"