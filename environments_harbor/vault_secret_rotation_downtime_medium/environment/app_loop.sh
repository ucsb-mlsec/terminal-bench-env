#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p /app/config /app/db /app/logs

# Infinite loop
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Read credentials from JSON file
    if [ ! -f /app/config/credentials.json ]; then
        echo "[$TIMESTAMP] FAILURE" >> /app/logs/auth.log
        sleep 3
        continue
    fi
    
    # Parse JSON to extract username and password
    USERNAME=$(jq -r '.username // empty' /app/config/credentials.json 2>/dev/null)
    PASSWORD=$(jq -r '.password // empty' /app/config/credentials.json 2>/dev/null)
    
    if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
        echo "[$TIMESTAMP] FAILURE" >> /app/logs/auth.log
        sleep 3
        continue
    fi
    
    # Check if username is 'dbuser'
    if [ "$USERNAME" != "dbuser" ]; then
        echo "[$TIMESTAMP] FAILURE" >> /app/logs/auth.log
        sleep 3
        continue
    fi
    
    # Read valid passwords
    if [ ! -f /app/db/valid_passwords.txt ]; then
        echo "[$TIMESTAMP] FAILURE" >> /app/logs/auth.log
        sleep 3
        continue
    fi
    
    # Check if password matches any valid password
    STATUS="FAILURE"
    while IFS= read -r valid_pass; do
        if [ "$PASSWORD" = "$valid_pass" ]; then
            STATUS="SUCCESS"
            break
        fi
    done < /app/db/valid_passwords.txt
    
    # Log the result
    echo "[$TIMESTAMP] $STATUS" >> /app/logs/auth.log
    
    # Sleep for 3 seconds
    sleep 3
done