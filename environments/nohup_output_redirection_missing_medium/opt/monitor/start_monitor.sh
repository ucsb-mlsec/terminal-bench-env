#!/bin/bash

# Start system monitoring script in background
# This script launches the continuous monitoring process using nohup

# Attempt to start monitor with output redirection
nohup /opt/monitor/system_check.sh > /var/log/monitor/system_check.log 2>&1 &

echo "Monitor started with PID $!"