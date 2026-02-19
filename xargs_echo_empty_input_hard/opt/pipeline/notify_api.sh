#!/bin/bash

# notify_api.sh
# This script sends API notifications for completed transactions
# It processes transaction files and notifies external systems

# Configuration
TRANSACTION_DIR="/var/data/transactions"
READY_DIR="/var/data/transactions/ready"
API_ENDPOINT="https://api.example.com/notify"

echo 'Sending API notifications for completed transactions'

# Send notifications to external API
# Find all completed transaction files and send them to the API
# This will trigger expensive API calls even with no completed transactions
grep -l "STATUS=COMPLETE" /var/data/transactions/*.txn 2>/dev/null | \
    xargs -I FILE curl -X POST https://api.example.com/notify -d @FILE

# Batch notify service
# Process files in the ready directory in batches of 5
# This batch processor will run even when the ready directory is empty
# causing unnecessary invocations of the batch notification service
find /var/data/transactions/ready/ -type f | \
    xargs -n 5 /opt/pipeline/batch_notify.py

# Log completion
echo "API notification process completed at $(date)"

# Exit successfully
exit 0