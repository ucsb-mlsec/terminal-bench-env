#!/bin/bash

# Generate reports from processed data
# This script aggregates data from multiple sources and creates summary reports

REPORT_DIR="/home/agent/reports/raw"
DATA_DIR="/home/agent/data"
OUTPUT_DIR="/home/agent/reports/output"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

echo "Starting report generation..."

# Process CSV reports - BROKEN: will fail if no CSV files exist
for report in /home/agent/reports/raw/*.csv
do
    echo "Processing CSV report: $report"
    line_count=$(wc -l < "$report")
    echo "Report has $line_count lines"
done

# Aggregate JSON data - BROKEN: will fail with ambiguous redirect if no JSON files
cat /home/agent/reports/raw/*.json | jq '.[]' > "$OUTPUT_DIR/summary.txt"

# Sort all text files - BROKEN: ambiguous redirect on empty glob
sort /home/agent/data/*.txt > "$OUTPUT_DIR/sorted.txt"

# Count log files - BROKEN: will fail on empty directory
files=(/home/agent/data/*.log)
echo "Found ${#files[@]} log files to process"

# Process all data files by type - BROKEN: multiple unmatched patterns
for datafile in /home/agent/data/*.dat /home/agent/data/*.tmp
do
    echo "Processing data file: $datafile"
    cp "$datafile" "$OUTPUT_DIR/"
done

# Archive old reports - BROKEN: will crash if no old reports exist
old_reports=(/home/agent/reports/raw/*.old)
for old in "${old_reports[@]}"
do
    echo "Archiving: $old"
    gzip "$old"
done

# Generate summary statistics - BROKEN: fails on empty pattern
cat /home/agent/data/processed/*.stats > "$OUTPUT_DIR/statistics.txt"

echo "Report generation complete"