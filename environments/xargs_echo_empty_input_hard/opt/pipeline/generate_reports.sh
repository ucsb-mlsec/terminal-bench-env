#!/bin/bash

# Generate daily transaction reports
# This script processes transaction files and generates PDF reports
# Runs nightly via cron to produce management summaries

echo "Generating transaction reports"
echo "Connecting to reporting database..."

# Set up report directories
REPORT_DIR="/var/reports/daily"
mkdir -p "$REPORT_DIR"

# Generate PDF reports from daily summaries
# Process all CSV files modified in the last 24 hours
find /var/data/transactions/ -name "*_daily.csv" -mtime -1 | xargs /opt/pipeline/report_generator --format=pdf

# Create consolidated reports
# Process each JSON summary file individually
ls /var/data/transactions/summary_*.json 2>/dev/null | xargs -n 1 python3 /opt/pipeline/create_report.py

# Archive old reports
find "$REPORT_DIR" -name "*.pdf" -mtime +30 -type f | xargs rm -f

# Update report index
echo "Updating report index..."

# Send notification on completion
echo "Report generation completed at $(date)"