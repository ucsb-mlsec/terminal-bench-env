#!/bin/bash

# Automated certificate renewal script for multi-domain infrastructure
# This script attempts to renew SSL certificates for all domains in the inventory
# Scheduled via cron: 0 2 * * *

# Configuration variables
CERT_INVENTORY='/opt/certs/cert_inventory.json'
LOG_DIR='/var/log/certbot'
CERTBOT_CMD='/usr/bin/certbot'
LOG_FILE="$LOG_DIR/certbot_renewal.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function to read domains from inventory JSON
get_domains_from_inventory() {
    if [ ! -f "$CERT_INVENTORY" ]; then
        log_message "ERROR: Certificate inventory file not found at $CERT_INVENTORY"
        exit 1
    fi
    
    # Extract domain names from JSON inventory
    # Assumes JSON structure with "domains" array containing domain objects
    domains=$(jq -r '.domains[].name' "$CERT_INVENTORY" 2>/dev/null)
    
    if [ -z "$domains" ]; then
        log_message "ERROR: No domains found in inventory"
        exit 1
    fi
    
    echo "$domains"
}

# Main renewal logic
log_message "===== Starting certificate renewal process ====="

# Get all domains from inventory
domains=$(get_domains_from_inventory)

# Loop through each domain and attempt renewal
for domain in $domains; do
    log_message "Processing domain: $domain"
    
    # Attempt certificate renewal with force-renewal flag
    # Using standalone mode for simplicity
    $CERTBOT_CMD certonly --standalone \
        --domain "$domain" \
        --non-interactive \
        --agree-tos \
        --force-renewal \
        --email admin@example.com \
        >> "$LOG_FILE" 2>&1
    
    # Check if renewal was successful
    if [ $? -ne 0 ]; then
        log_message "ERROR: Failed to renew certificate for $domain"
    else
        log_message "SUCCESS: Certificate renewed for $domain"
    fi
done

log_message "===== Certificate renewal process completed ====="

# Exit successfully
exit 0