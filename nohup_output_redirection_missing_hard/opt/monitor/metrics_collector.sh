#!/bin/bash

while true; do
    echo "=========================================="
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"
    echo "System Uptime and Load:"
    uptime
    echo ""
    echo "Memory Usage:"
    free -m
    echo ""
    echo "Disk Usage:"
    df -h /
    echo ""
    echo "CPU Info:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU Usage: " 100 - $1"%"}'
    echo "=========================================="
    echo ""
    sleep 8
done