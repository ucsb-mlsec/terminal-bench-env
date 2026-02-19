#!/bin/bash

# process_refunds.sh
# Processing refund transactions from the financial data pipeline
# This script handles both individual refund files and batch processing

# Initialize logging
LOG_FILE="/var/log/pipeline/refunds.log"

echo "Initializing refund processing system..."
echo "[$(date)] Starting refund processing" >> "$LOG_FILE"

# Check if transaction directory exists
if [ ! -d "/var/data/transactions" ]; then
    echo "Error: Transaction directory not found"
    exit 1
fi

echo "Processing refund transactions"

# Process all pending refund transactions
# Find all transaction files containing REFUND status
# Each file is processed individually to maintain transaction integrity
grep -l "REFUND" /var/data/transactions/*.txn 2>/dev/null | xargs -I {} /opt/pipeline/process_refund {}

# Batch process refunds
# Process pending refund files in batches of 10 for efficiency
# This reduces database connection overhead
find /var/data/transactions/refunds/ -name "*.pending" | xargs -n 10 /usr/bin/refund-processor --batch-mode

# Log completion
echo "[$(date)] Refund processing completed" >> "$LOG_FILE"
echo "Refund processing completed successfully"

exit 0