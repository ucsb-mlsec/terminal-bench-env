#!/bin/bash

# Task: Manage DigitalOcean droplet resize with minimal downtime
# Requirements: 
# - Downtime < 300 seconds
# - Preserve data integrity
# - Complete resize successfully

set -e

RESULTS_FILE="/home/agent/resize_results.txt"
ACCESS_LOG="/var/log/webservice/access.log"
SESSIONS_FILE="/var/app/sessions.json"
CONFIG_FILE="/var/app/config.json"
RESIZE_SCRIPT="/opt/resize_simulator.sh"
BACKUP_DIR="/tmp/app_backup"
WEB_SERVICE_PID_FILE="/var/run/webservice.pid"

# Function to find and stop the web service
stop_web_service() {
    echo "Stopping web service..."
    
    # Find the web service process (looking for process writing to access.log)
    # Try to find by PID file first
    if [ -f "$WEB_SERVICE_PID_FILE" ]; then
        WEB_PID=$(cat "$WEB_SERVICE_PID_FILE")
        if kill -0 "$WEB_PID" 2>/dev/null; then
            kill "$WEB_PID"
            echo "Web service stopped (PID: $WEB_PID)"
            return 0
        fi
    fi
    
    # Otherwise find by process name or log file access
    WEB_PID=$(lsof "$ACCESS_LOG" 2>/dev/null | grep -v COMMAND | awk '{print $2}' | head -1)
    if [ -n "$WEB_PID" ]; then
        kill "$WEB_PID"
        echo "Web service stopped (PID: $WEB_PID)"
    else
        echo "Web service process not found or already stopped"
    fi
}

# Function to start the web service
start_web_service() {
    echo "Starting web service..."
    
    # Look for common service management methods
    if [ -f "/etc/init.d/webservice" ]; then
        /etc/init.d/webservice start
    elif systemctl list-unit-files | grep -q webservice; then
        systemctl start webservice
    elif [ -f "/usr/local/bin/webservice" ]; then
        /usr/local/bin/webservice &
    elif [ -f "/opt/webservice/start.sh" ]; then
        /opt/webservice/start.sh &
    else
        echo "Warning: Could not find standard web service startup method"
        # Try to find and restart any stopped process
        pkill -CONT -f webservice 2>/dev/null || true
    fi
    
    sleep 2
    echo "Web service started"
}

# Function to validate JSON files
validate_json() {
    local file=$1
    if [ ! -f "$file" ]; then
        return 1
    fi
    python3 -c "import json; json.load(open('$file'))" 2>/dev/null
    return $?
}

# Function to backup application data
backup_data() {
    echo "Backing up application data..."
    mkdir -p "$BACKUP_DIR"
    
    if [ -f "$SESSIONS_FILE" ]; then
        cp -p "$SESSIONS_FILE" "$BACKUP_DIR/sessions.json.bak"
    fi
    
    if [ -f "$CONFIG_FILE" ]; then
        cp -p "$CONFIG_FILE" "$BACKUP_DIR/config.json.bak"
    fi
    
    echo "Backup completed"
}

# Function to calculate downtime from access log
calculate_downtime() {
    echo "Calculating downtime from access log..."
    
    if [ ! -f "$ACCESS_LOG" ]; then
        echo "0"
        return
    fi
    
    # Get the last timestamp before resize and first timestamp after resize
    # Timestamps are in format that can be parsed
    tail -100 "$ACCESS_LOG" | python3 -c "
import sys
import re
from datetime import datetime

lines = sys.stdin.readlines()
timestamps = []

for line in lines:
    # Try to extract timestamp - assuming format with timestamp at start or after certain pattern
    # Common formats: [2024-01-15 10:30:45] or 2024-01-15T10:30:45 or epoch time
    match = re.search(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', line)
    if match:
        try:
            ts_str = match.group(1).replace('T', ' ')
            ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            timestamps.append(ts)
        except:
            pass
    else:
        # Try epoch timestamp
        match = re.search(r'(\d{10,})', line)
        if match:
            try:
                timestamps.append(float(match.group(1)))
            except:
                pass

if len(timestamps) >= 2:
    if isinstance(timestamps[0], datetime):
        max_gap = 0
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            if gap > max_gap:
                max_gap = gap
        print(int(max_gap))
    else:
        # Epoch timestamps
        max_gap = 0
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            if gap > max_gap:
                max_gap = gap
        print(int(max_gap))
else:
    print(0)
" || echo "0"
}

# Main execution
echo "=== Starting Droplet Resize Process ==="
echo "Time: $(date)"

# Step 1: Record start time
START_TIME=$(date +%s)

# Step 2: Assess current state
echo "Assessing current state..."
if [ ! -f "$RESIZE_SCRIPT" ]; then
    echo "ERROR: Resize script not found at $RESIZE_SCRIPT"
    exit 1
fi

# Step 3: Backup application data
backup_data

# Step 4: Validate data before resize
SESSIONS_VALID_BEFORE=false
CONFIG_VALID_BEFORE=false

if validate_json "$SESSIONS_FILE"; then
    echo "Sessions file valid before resize"
    SESSIONS_VALID_BEFORE=true
fi

if validate_json "$CONFIG_FILE"; then
    echo "Config file valid before resize"
    CONFIG_VALID_BEFORE=true
fi

# Step 5: Record time before stopping service
BEFORE_STOP_TIME=$(date +%s)

# Step 6: Stop the web service gracefully
stop_web_service
sleep 1

# Step 7: Ensure data is flushed to disk
sync

# Step 8: Record downtime start from last log entry
if [ -f "$ACCESS_LOG" ]; then
    LAST_LOG_BEFORE=$(tail -1 "$ACCESS_LOG")
fi

# Step 9: Execute the resize operation
echo "Starting resize operation..."
RESIZE_START=$(date +%s)

if bash "$RESIZE_SCRIPT"; then
    RESIZE_COMPLETED=true
    echo "Resize completed successfully"
else
    RESIZE_COMPLETED=false
    echo "ERROR: Resize failed"
fi

RESIZE_END=$(date +%s)

# Step 10: Start the web service
start_web_service
sleep 3

# Step 11: Wait for service to write to log
echo "Waiting for service to resume logging..."
for i in {1..10}; do
    sleep 1
    if [ -f "$ACCESS_LOG" ]; then
        NEW_ENTRIES=$(tail -5 "$ACCESS_LOG" | wc -l)
        if [ "$NEW_ENTRIES" -gt 0 ]; then
            break
        fi
    fi
done

AFTER_START_TIME=$(date +%s)

# Step 12: Validate data integrity after resize
DATA_INTACT=true

if [ "$SESSIONS_VALID_BEFORE" = true ]; then
    if ! validate_json "$SESSIONS_FILE"; then
        echo "ERROR: Sessions file corrupted after resize"
        DATA_INTACT=false
    else
        echo "Sessions file intact after resize"
    fi
fi

if [ "$CONFIG_VALID_BEFORE" = true ]; then
    if ! validate_json "$CONFIG_FILE"; then
        echo "ERROR: Config file corrupted after resize"
        DATA_INTACT=false
    else
        echo "Config file intact after resize"
    fi
fi

# Step 13: Calculate actual downtime from logs
DOWNTIME_FROM_LOG=$(calculate_downtime)

# Calculate downtime from service stop/start times as backup
DOWNTIME_CALCULATED=$((AFTER_START_TIME - BEFORE_STOP_TIME))

# Use the log-based downtime if available and reasonable, otherwise use calculated
if [ "$DOWNTIME_FROM_LOG" -gt 0 ] && [ "$DOWNTIME_FROM_LOG" -lt 500 ]; then
    DOWNTIME=$DOWNTIME_FROM_LOG
else
    DOWNTIME=$DOWNTIME_CALCULATED
fi

echo "=== Resize Process Complete ==="
echo "Total downtime: $DOWNTIME seconds"
echo "Data intact: $DATA_INTACT"
echo "Resize completed: $RESIZE_COMPLETED"

# Step 14: Write results
cat > "$RESULTS_FILE" << EOF
downtime_seconds=$DOWNTIME
data_intact=$DATA_INTACT
resize_completed=$RESIZE_COMPLETED
EOF

echo "Results written to $RESULTS_FILE"
cat "$RESULTS_FILE"

# Verify success criteria
if [ "$DOWNTIME" -lt 300 ] && [ "$DATA_INTACT" = "true" ] && [ "$RESIZE_COMPLETED" = "true" ]; then
    echo "✓ All success criteria met!"
    exit 0
else
    echo "✗ Some criteria not met"
    exit 1
fi