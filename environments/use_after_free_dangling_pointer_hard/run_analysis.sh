#!/bin/bash
set -e

echo "Building log analyzer..."

# Clean and build
make clean
make all

# Check if build succeeded
if [ ! -f "./log_analyzer" ]; then
    echo "Error: Build failed, log_analyzer executable not found"
    exit 1
fi

# Create output directory
mkdir -p results

# Process all log files
for logfile in logs/*.log; do
    if [ -f "$logfile" ]; then
        filename=$(basename "$logfile")
        statsfile="results/${filename%.log}.stats"
        echo "Processing $filename..."
        
        if ! ./log_analyzer "$logfile" > "$statsfile"; then
            echo "Error: Failed to process $filename"
            exit 1
        fi
    fi
done

echo "All log files processed successfully!"
exit 0