#!/bin/bash
# System Monitoring Script
# This script continuously monitors system resources

# Log file location
LOG_DIR="/var/log/monitor"
LOG_FILE="$LOG_DIR/system_check.log"

# Redirect output to log file
exec > $LOG_FILE 2>&1 &

echo "=== System Monitor Starting ==="
echo "Monitor initialized at $(date)"

# Main monitoring loop
while true; do
    echo ""
    echo "----------------------------------------"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    
    echo ""
    echo "=== System Uptime and Load ==="
    uptime
    
    echo ""
    echo "=== Memory Usage ==="
    free -m | grep -E "Mem|Swap"
    
    echo ""
    echo "=== Disk Usage ==="
    df -h / | grep -v Filesystem
    
    echo ""
    echo "=== CPU Information ==="
    top -bn1 | grep "Cpu(s)" | head -1
    
    # Sleep interval between checks
    sleep 8
done