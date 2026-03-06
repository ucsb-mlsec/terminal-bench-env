#!/bin/bash

# import_to_db.sh
# Batch database import script for processed transaction files
# This script handles the database import phase of the transaction pipeline

# Import all processed transaction files to production database
# Files must be validated and processed before import

set -e

TRANSACTION_DIR="/var/data/transactions"
LOG_FILE="/var/log/pipeline/db_import.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log start of import batch
echo "Starting database import batch" | tee -a "$LOG_FILE"
echo "Import started at: $(date)" >> "$LOG_FILE"

# Phase 1: Import processed CSV files
# These files have been validated and are ready for database insertion
echo "Phase 1: Importing processed CSV files..." | tee -a "$LOG_FILE"

find /var/data/transactions/ -name "*_processed.csv" | xargs -n 1 /opt/pipeline/db_import_tool

echo "Processed CSV imports complete" >> "$LOG_FILE"

# Phase 2: Batch import of .import files
# These contain bulk transaction data for the database loader
echo "Phase 2: Running batch database loader..." | tee -a "$LOG_FILE"

ls /var/data/transactions/*.import 2>/dev/null | xargs /usr/bin/database-loader --batch

echo "Batch import complete" >> "$LOG_FILE"

# Final status
echo "Database import batch completed at: $(date)" | tee -a "$LOG_FILE"
echo "All transactions imported successfully" >> "$LOG_FILE"

exit 0