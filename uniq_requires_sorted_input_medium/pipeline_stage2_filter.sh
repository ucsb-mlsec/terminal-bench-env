#!/bin/bash

# Stage 2: Filter Pipeline
# Purpose: Filter out invalid and private IP addresses
# Input: /opt/analytics/data/output/stage1_ips.txt
# Output: /opt/analytics/data/output/stage2_filtered.txt

# Define input and output paths
INPUT_FILE="/opt/analytics/data/output/stage1_ips.txt"
OUTPUT_FILE="/opt/analytics/data/output/stage2_filtered.txt"

# Create output directory if it doesn't exist
mkdir -p /opt/analytics/data/output

# Clear output file if it exists
> "$OUTPUT_FILE"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found" >&2
    exit 1
fi

# Process each line from the input file
while IFS= read -r ip || [ -n "$ip" ]; do
    # Skip empty lines
    if [ -z "$ip" ]; then
        continue
    fi
    
    # Trim whitespace
    ip=$(echo "$ip" | tr -d '[:space:]')
    
    # Skip if still empty after trimming
    if [ -z "$ip" ]; then
        continue
    fi
    
    # Validate IP address format (basic IPv4 pattern)
    # Must match: 0-255.0-255.0-255.0-255
    if ! echo "$ip" | grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}$'; then
        continue
    fi
    
    # Additional validation: each octet must be 0-255
    valid=true
    IFS='.' read -ra OCTETS <<< "$ip"
    for octet in "${OCTETS[@]}"; do
        if [ "$octet" -gt 255 ] || [ "$octet" -lt 0 ]; then
            valid=false
            break
        fi
    done
    
    if [ "$valid" = false ]; then
        continue
    fi
    
    # Filter out localhost (127.0.0.0/8)
    if echo "$ip" | grep -Eq '^127\.'; then
        continue
    fi
    
    # Filter out private IP ranges
    # 10.0.0.0/8
    if echo "$ip" | grep -Eq '^10\.'; then
        continue
    fi
    
    # 172.16.0.0/12 (172.16.0.0 to 172.31.255.255)
    if echo "$ip" | grep -Eq '^172\.(1[6-9]|2[0-9]|3[0-1])\.'; then
        continue
    fi
    
    # 192.168.0.0/16
    if echo "$ip" | grep -Eq '^192\.168\.'; then
        continue
    fi
    
    # Filter out link-local addresses (169.254.0.0/16)
    if echo "$ip" | grep -Eq '^169\.254\.'; then
        continue
    fi
    
    # Filter out multicast addresses (224.0.0.0/4)
    if echo "$ip" | grep -Eq '^(22[4-9]|23[0-9])\.'; then
        continue
    fi
    
    # Filter out broadcast address
    if [ "$ip" = "255.255.255.255" ]; then
        continue
    fi
    
    # Filter out 0.0.0.0
    if [ "$ip" = "0.0.0.0" ]; then
        continue
    fi
    
    # If we made it here, the IP is valid and public
    echo "$ip" >> "$OUTPUT_FILE"
    
done < "$INPUT_FILE"

# Log completion
echo "Stage 2 filtering complete. Output written to $OUTPUT_FILE" >&2