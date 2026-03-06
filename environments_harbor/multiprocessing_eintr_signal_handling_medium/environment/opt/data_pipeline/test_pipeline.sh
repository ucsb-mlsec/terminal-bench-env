#!/bin/bash
set -e

# Test script for data processing pipeline under signal stress conditions
# This script simulates production interruptions by sending signals while processing

echo "Starting pipeline test with signal stress conditions..."

# Clear any previous test results
rm -f /tmp/processed_count.txt

# Start the processor in background with 100 items
python /opt/data_pipeline/processor.py 100 &
PID=$!

echo "Pipeline started with PID: $PID"

# Give the process a moment to start up
sleep 0.2

# Send multiple signals to simulate production interruptions
echo "Sending SIGTERM signal..."
kill -TERM $PID 2>/dev/null || true
sleep 0.3

echo "Sending SIGUSR1 signal..."
kill -USR1 $PID 2>/dev/null || true
sleep 0.4

echo "Sending another SIGTERM signal..."
kill -TERM $PID 2>/dev/null || true
sleep 0.3

echo "Sending SIGUSR1 signal..."
kill -USR1 $PID 2>/dev/null || true
sleep 0.2

echo "Sending final SIGTERM signal..."
kill -TERM $PID 2>/dev/null || true

# Wait for the process to complete
echo "Waiting for pipeline to complete..."
wait $PID
EXIT_CODE=$?

echo "Pipeline exited with code: $EXIT_CODE"

# Verify results
if [ ! -f /tmp/processed_count.txt ]; then
    echo "Test FAILED: /tmp/processed_count.txt does not exist"
    exit 1
fi

PROCESSED_COUNT=$(cat /tmp/processed_count.txt)
echo "Processed count: $PROCESSED_COUNT"

if [ "$PROCESSED_COUNT" -eq 100 ]; then
    echo "Test PASSED: All 100 items processed successfully"
    exit 0
else
    echo "Test FAILED: Expected 100 items, got $PROCESSED_COUNT"
    exit 1
fi