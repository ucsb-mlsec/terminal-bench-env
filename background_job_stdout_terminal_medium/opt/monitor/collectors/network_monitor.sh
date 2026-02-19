#!/bin/bash

# Network latency monitor - checks ping times to various endpoints

while true; do
    echo "=== $(date) ==="
    echo "Network Monitor: Checking latency..."
    
    # Simulate ping to Google DNS
    LATENCY1=$((20 + RANDOM % 20))
    echo "Ping to 8.8.8.8: ${LATENCY1} ms"
    
    # Simulate ping to Cloudflare DNS
    LATENCY2=$((15 + RANDOM % 25))
    echo "Ping to 1.1.1.1: ${LATENCY2} ms"
    
    echo ""
    
    # Wait 5 seconds before next check
    sleep 5
done