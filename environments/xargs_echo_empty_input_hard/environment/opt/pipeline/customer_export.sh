#!/bin/bash
# Export customer transaction data for analysis
# This script processes customer transaction files and exports them
# to the analysis platform for reporting and compliance

set -e

echo "Starting customer data export process"
echo "========================================"

# Export customer transaction data for analysis
echo "Exporting customer data"

# Find and process recent customer transaction files
# Concatenate, sort, and deduplicate customer data from the last 7 days
find /var/data/transactions/ -name "customer_*.csv" -mtime -7 | xargs cat | sort | uniq > /var/exports/customers.txt

echo "Customer data exported to /var/exports/customers.txt"

# Process VIP customer transactions separately with encryption
echo "Connecting to export service..."
echo "Processing VIP transactions..."

# Export VIP transactions using encrypted export method
ls /var/data/transactions/vip_*.txn 2>/dev/null | xargs /opt/pipeline/vip_exporter --encrypted

echo "VIP customer data exported successfully"
echo "Export process completed"