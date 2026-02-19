#!/bin/bash

set -e

PROCESSOR_DIR=/opt/dataprocessor
OUTPUT_DIR=/opt/dataprocessor/output
SOLUTION_FILE=/solution/fix_summary.json
START_TIME=$(date +%s)

echo "=== Data Processor Test Script ==="
echo "Starting at $(date)"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p /solution

# Clean any previous results
rm -f $OUTPUT_DIR/results.json
rm -f $OUTPUT_DIR/*.log

# Start the processor in background
echo "Starting processor..."
cd $PROCESSOR_DIR
python3 processor.py > $OUTPUT_DIR/processor.log 2>&1 &
PROCESSOR_PID=$!

echo "Processor started with PID: $PROCESSOR_PID"

# Signal sender function
send_signals() {
    echo "Starting signal sender..."
    while kill -0 $PROCESSOR_PID 2>/dev/null; do
        # Random interval between 2-5 seconds
        INTERVAL=$(shuf -i 2-5 -n 1)
        sleep $INTERVAL
        
        # Send SIGUSR1 to main process
        if kill -0 $PROCESSOR_PID 2>/dev/null; then
            kill -USR1 $PROCESSOR_PID 2>/dev/null || true
            
            # Also send to all child processes
            pkill -USR1 -P $PROCESSOR_PID 2>/dev/null || true
            
            echo "Sent SIGUSR1 signal at $(date +%s)"
        fi
    done
    echo "Signal sender stopped (processor finished)"
}

# Start signal sender in background
send_signals &
SIGNAL_PID=$!

# Wait for processor to complete
echo "Waiting for processor to complete..."
wait $PROCESSOR_PID
PROCESSOR_EXIT_CODE=$?

# Stop signal sender
kill $SIGNAL_PID 2>/dev/null || true
wait $SIGNAL_PID 2>/dev/null || true

END_TIME=$(date +%s)
PROCESSING_TIME=$((END_TIME - START_TIME))

echo "Processor completed with exit code: $PROCESSOR_EXIT_CODE"
echo "Processing time: $PROCESSING_TIME seconds"

# Validation checks
TEST_PASSED=true
WORKERS_CRASHED=0

echo "=== Validation Checks ==="

# Check if results.json exists
if [ ! -f "$OUTPUT_DIR/results.json" ]; then
    echo "ERROR: results.json not found"
    TEST_PASSED=false
else
    echo "✓ results.json exists"
    
    # Verify all 4 CSV files were processed
    FILES_PROCESSED=$(grep -c '"file":' $OUTPUT_DIR/results.json || echo 0)
    if [ "$FILES_PROCESSED" -ne 4 ]; then
        echo "ERROR: Expected 4 files processed, found $FILES_PROCESSED"
        TEST_PASSED=false
    else
        echo "✓ All 4 files processed"
    fi
    
    # Count total records (should be 400)
    TOTAL_RECORDS=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/results.json')); print(sum(f['records'] for f in data['files']))" 2>/dev/null || echo 0)
    if [ "$TOTAL_RECORDS" -ne 400 ]; then
        echo "ERROR: Expected 400 records, found $TOTAL_RECORDS"
        TEST_PASSED=false
    else
        echo "✓ All 400 records processed"
    fi
    
    # Check for data corruption
    CHECKSUM_OK=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/results.json')); print('true' if all(f.get('checksum_valid', False) for f in data['files']) else 'false')" 2>/dev/null || echo "false")
    if [ "$CHECKSUM_OK" != "true" ]; then
        echo "ERROR: Data corruption detected"
        TEST_PASSED=false
    else
        echo "✓ No data corruption detected"
    fi
fi

# Count worker crashes from logs
if [ -f "$OUTPUT_DIR/processor.log" ]; then
    WORKERS_CRASHED=$(grep -c "Worker crashed\|EINTR\|Interrupted system call\|Worker.*died" $OUTPUT_DIR/processor.log 2>/dev/null || echo 0)
    if [ "$WORKERS_CRASHED" -gt 0 ]; then
        echo "WARNING: $WORKERS_CRASHED worker crashes detected"
    else
        echo "✓ No worker crashes detected"
    fi
fi

# Check processor exit code
if [ "$PROCESSOR_EXIT_CODE" -ne 0 ]; then
    echo "ERROR: Processor exited with non-zero code: $PROCESSOR_EXIT_CODE"
    TEST_PASSED=false
fi

# Check processing time
if [ "$PROCESSING_TIME" -gt 900 ]; then
    echo "WARNING: Processing took longer than 15 minutes"
    TEST_PASSED=false
else
    echo "✓ Processing completed within time limit"
fi

# Get list of modified files (placeholder - will be filled by actual solution)
FILES_MODIFIED='[]'
if [ -f "$PROCESSOR_DIR/.modified_files" ]; then
    FILES_MODIFIED=$(cat "$PROCESSOR_DIR/.modified_files")
fi

# Create fix_summary.json
cat > $SOLUTION_FILE << EOF
{
  "files_modified": $FILES_MODIFIED,
  "test_passed": $TEST_PASSED,
  "processing_time_seconds": $PROCESSING_TIME,
  "workers_crashed": $WORKERS_CRASHED
}
EOF

echo ""
echo "=== Test Results ==="
echo "Test Passed: $TEST_PASSED"
echo "Processing Time: $PROCESSING_TIME seconds"
echo "Worker Crashes: $WORKERS_CRASHED"
echo "Summary saved to: $SOLUTION_FILE"

if [ "$TEST_PASSED" = true ]; then
    echo "✓ TEST PASSED"
    exit 0
else
    echo "✗ TEST FAILED"
    exit 1
fi