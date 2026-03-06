#!/bin/bash

# Audit Logger Script
# This script processes transaction files and creates audit log entries
# This creates expensive audit database entries

echo "Starting audit logging process..."

# Log all processed transactions to audit system
# This operation connects to the audit database and creates entries
# Each entry triggers database writes and compliance logging
echo "Creating audit log entries"
echo "Opening audit database connection..."

# Process all files that were processed in the last 24 hours
# Find recently processed transaction files and log them
find /var/data/transactions/ -name "*.processed" -mtime -1 | xargs -I {} /opt/pipeline/log_audit {}

echo "Processed files logged to audit system"

# High-value transaction audit
# Special audit requirements for high-value transactions
# These require additional compliance documentation
echo "Scanning for high-value transactions..."
echo "Opening audit database connection..."

# Search for high-value transactions and create audit entries
# Each audit entry is expensive and triggers compliance workflows
grep -l "HIGH_VALUE" /var/data/transactions/*.txn 2>/dev/null | xargs -n 1 python3 /opt/pipeline/audit_entry.py

echo "High-value transaction audit complete"
echo "All audit database connections closed"
echo "Audit logging process finished"