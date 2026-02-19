#!/bin/bash

# Create the log directory
mkdir -p /var/log/application

# Array of node IDs
nodes=("node001" "node002" "node003" "node004" "node005")

# Log levels and sample messages
log_levels=("INFO" "WARN" "ERROR" "DEBUG")
messages=(
    "Application started successfully"
    "Processing request from client"
    "Database connection established"
    "Cache miss for key"
    "User authentication successful"
    "Configuration loaded from file"
    "Request processing completed"
    "Memory usage: 45%"
    "Network latency detected"
    "Service health check passed"
    "Failed to connect to remote service"
    "Timeout waiting for response"
    "Invalid input received from user"
    "Permission denied for operation"
    "Resource not found"
    "Critical system error occurred"
)

# Generate 1500 test log files
echo "Generating test log files..."

for i in $(seq 1 1500); do
    # Random date between 90 days ago and today
    days_ago=$((RANDOM % 91))
    
    # Generate timestamp
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS date command
        date_part=$(date -v-${days_ago}d +%Y%m%d)
    else
        # Linux date command
        date_part=$(date -d "${days_ago} days ago" +%Y%m%d)
    fi
    
    # Random time
    hour=$(printf "%02d" $((RANDOM % 24)))
    minute=$(printf "%02d" $((RANDOM % 60)))
    second=$(printf "%02d" $((RANDOM % 60)))
    time_part="${hour}${minute}${second}"
    
    # Random node
    node_id=${nodes[$((RANDOM % ${#nodes[@]}))]}
    
    # Create filename
    filename="/var/log/application/app_${date_part}_${time_part}_${node_id}.log"
    
    # Number of log lines (5-10)
    num_lines=$((5 + RANDOM % 6))
    
    # Decide if this file should contain errors (30% chance)
    has_error=$((RANDOM % 100))
    
    # Generate log content
    for line in $(seq 1 $num_lines); do
        # Random log level
        if [ $has_error -lt 30 ] && [ $line -eq $((num_lines / 2)) ]; then
            # Force an ERROR in this line
            level="ERROR"
        else
            level=${log_levels[$((RANDOM % ${#log_levels[@]}))]}
        fi
        
        # Random message
        message=${messages[$((RANDOM % ${#messages[@]}))]}
        
        # Generate timestamp for log entry
        log_timestamp=$(date -d "${days_ago} days ago" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -v-${days_ago}d "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date "+%Y-%m-%d %H:%M:%S")
        
        # Write log entry
        echo "[$log_timestamp] [$level] [$node_id] $message" >> "$filename"
    done
    
    # Progress indicator
    if [ $((i % 100)) -eq 0 ]; then
        echo "Created $i files..."
    fi
done

echo "Test log generation complete. Created 1500 log files in /var/log/application"
echo "Files range from 90 days ago to today, approximately 30% contain ERROR entries"
