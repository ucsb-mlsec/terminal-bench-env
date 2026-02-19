#!/bin/bash
# disk_report.sh - Generates disk usage report for log files
# This script calculates total disk space used by application log files
# WARNING: This version has argument list limitations with large number of files

LOG_DIR="/var/log/application"

echo "============================================"
echo "Disk Usage Report for Application Logs"
echo "============================================"
echo ""
echo "Calculating total size of log files..."
echo ""

# Problematic approach: passing all files as arguments at once
# This will fail with "Argument list too long" when there are too many files
TOTAL_SIZE=$(du -ch $LOG_DIR/*.log 2>/dev/null | tail -1 | awk '{print $1}')

if [ $? -eq 0 ]; then
    echo "Total disk usage: $TOTAL_SIZE"
else
    echo "Error: Failed to calculate disk usage"
    exit 1
fi

echo ""
echo "Detailed breakdown:"
ls -lh $LOG_DIR/*.log | awk '{sum+=$5; print $9, $5} END {print "Total files:", NR}'

echo ""
echo "Report generated at: $(date)"