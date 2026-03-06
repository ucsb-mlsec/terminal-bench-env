#!/bin/bash

# Monitoring system - runs background data collection jobs
# This script launches three monitoring components as background processes

# Create log directory if it doesn't exist
LOG_DIR="/var/log/monitors"
mkdir -p "$LOG_DIR"

# Launch network latency monitor in background
# Redirect stdout and stderr to log file to prevent terminal corruption
/opt/monitor/collectors/network_monitor.sh > "$LOG_DIR/network_monitor.log" 2>&1 &

# Launch disk usage tracker in background
# Redirect stdout and stderr to log file to prevent terminal corruption
/opt/monitor/collectors/disk_monitor.sh > "$LOG_DIR/disk_monitor.log" 2>&1 &

# Launch log analyzer in background
# Redirect stdout and stderr to log file to prevent terminal corruption
/opt/monitor/collectors/log_monitor.sh > "$LOG_DIR/log_monitor.log" 2>&1 &

echo "All monitors started in background"
echo "Monitor output saved to: $LOG_DIR/"
echo "  - network_monitor.log"
echo "  - disk_monitor.log"
echo "  - log_monitor.log"