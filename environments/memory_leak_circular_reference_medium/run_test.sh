#!/bin/bash

# Test script to monitor memory usage of data_processor.py
# This helps verify the memory leak before and after the fix

echo "Starting data processor with memory monitoring..."
echo "Initial memory usage:"

# Run the Python script in background and capture its PID
python3 /workspace/data_processor.py &
PID=$!

# Monitor memory usage every second
echo "Monitoring memory usage (RSS in MB)..."
for i in {1..15}; do
    if ps -p $PID > /dev/null 2>&1; then
        # Get RSS (Resident Set Size) in KB and convert to MB
        MEM=$(ps -o rss= -p $PID | awk '{print $1/1024}')
        echo "Time ${i}s: ${MEM} MB"
        sleep 1
    else
        break
    fi
done

# Wait for process to complete
wait $PID

echo "Process completed"
echo "Expected: ~500MB before fix, <60MB after fix"