#!/bin/bash

# Initialize counters
TOTAL_REQUESTS=0
SUCCESS_COUNT=0
CLIENT_ERRORS=0
SERVER_ERRORS=0

# Process all log files
cat /var/log/access/*.log | while read line; do
    # Increment total requests
    ((TOTAL_REQUESTS++))
    
    # Extract HTTP status code (assuming standard Apache/nginx log format)
    # Status code is typically the 9th field in common log format
    STATUS=$(echo "$line" | awk '{print $9}')
    
    # Count by status code category
    if [[ "$STATUS" == "200" ]]; then
        ((SUCCESS_COUNT++))
    elif [[ "$STATUS" =~ ^4[0-9][0-9]$ ]]; then
        ((CLIENT_ERRORS++))
    elif [[ "$STATUS" =~ ^5[0-9][0-9]$ ]]; then
        ((SERVER_ERRORS++))
    fi
done

# Output results
echo "TOTAL_REQUESTS=$TOTAL_REQUESTS"
echo "SUCCESS_COUNT=$SUCCESS_COUNT"
echo "CLIENT_ERRORS=$CLIENT_ERRORS"
echo "SERVER_ERRORS=$SERVER_ERRORS"