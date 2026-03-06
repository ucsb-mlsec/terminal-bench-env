#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p /var/log/webservice

# Array of endpoints to simulate
endpoints=("/api/status" "/api/data" "/api/users" "/api/config" "/api/health")
methods=("GET" "POST" "GET" "GET" "GET")
status_codes=(200 200 201 200 200)

while true; do
    # Select random endpoint
    index=$((RANDOM % ${#endpoints[@]}))
    endpoint=${endpoints[$index]}
    method=${methods[$index]}
    status=${status_codes[$index]}
    
    # Generate random response time (10-99ms)
    response_time=$((10 + RANDOM % 90))
    
    # Write log entry with timestamp
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $method $endpoint $status ${response_time}ms" >> /var/log/webservice/access.log
    
    # Wait 2 seconds before next entry
    sleep 2
done