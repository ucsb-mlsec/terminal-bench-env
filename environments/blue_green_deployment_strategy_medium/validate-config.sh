#!/bin/bash

# Broken validation script for deployment configurations

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config-file>"
    exit 1
fi

CONFIG_FILE=$1

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Source the config file
source $CONFIG_FILE

# Check required keys exist
REQUIRED_KEYS="PORT HEALTH_ENDPOINT MAX_CONNECTIONS TIMEOUT_SECONDS"
for key in $REQUIRED_KEYS; do
    if [ -z ${!key} ]; then
        echo "Missing required key: $key"
        exit 1
    fi
done

# Validate PORT (buggy: using = instead of -eq, wrong range check)
if [ "$PORT" = "" ] && [ $PORT -lt 1 ] && [ $PORT -gt 65535 ]; then
    echo "PORT must be a number between 1 and 65535"
    exit 1
fi

# Check if PORT is numeric (buggy: missing quotes, wrong regex)
if ! [[ $PORT =~ ^[0-9]$ ]]; then
    echo "PORT must be numeric"
    exit 1
fi

# Validate MAX_CONNECTIONS (buggy: using string comparison)
if [ "$MAX_CONNECTIONS" = "0" ] || [ $MAX_CONNECTIONS -lt 0 ]; then
    echo "MAX_CONNECTIONS must be positive"
    exit 1
fi

# Validate TIMEOUT_SECONDS (buggy: wrong operator, missing quote)
if [ $TIMEOUT_SECONDS = 0 ]; then
    echo "TIMEOUT_SECONDS must be positive"
    exit 1
fi

# Validate HEALTH_ENDPOINT (buggy: case sensitivity, wrong substring check)
FIRST_CHAR=$(echo $HEALTH_ENDPOINT | cut -c1-1)
if [ $FIRST_CHAR = "/" ]; then
    echo "HEALTH_ENDPOINT must start with /"
    exit 1
fi

# If we got here, config is valid
echo "Configuration is valid"
exit 0