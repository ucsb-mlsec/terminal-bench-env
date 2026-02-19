#!/bin/bash

# CPU Monitor - checks system load
while true; do
  echo "[CPU Monitor] $(date): Load average: $(uptime | awk -F'load average:' '{print $2}')"
  sleep 3
done &

# Memory Monitor - checks memory usage
while true; do
  echo "[Memory Monitor] $(date): Memory usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
  sleep 4
done &

# Disk Monitor - checks disk space
while true; do
  echo "[Disk Monitor] $(date): Root disk usage: $(df -h / | tail -1 | awk '{print $5 " used"}')"
  sleep 5
done &

# Network Monitor - checks network connections
while true; do
  echo "[Network Monitor] $(date): Active connections: $(netstat -an 2>/dev/null | grep ESTABLISHED | wc -l)"
  sleep 3
done &

# Process Monitor - checks running processes
while true; do
  echo "[Process Monitor] $(date): Total processes: $(ps aux | wc -l)"
  sleep 4
done &

echo "Monitoring processes started"