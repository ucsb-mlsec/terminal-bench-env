#!/bin/bash

# test_concurrent_access.sh - Tests concurrent access to SQLite database

DB_PATH="/workspace/data/metrics.db"
SCRIPTS_DIR="/workspace/scripts"
NUM_PROCESSES=5
ITERATIONS=15
ERROR_LOG="/tmp/db_errors.log"

# Clean up previous error log
> "$ERROR_LOG"

echo "Testing concurrent database access..."
echo "Processes: $NUM_PROCESSES"
echo "Iterations per process: $ITERATIONS"
echo "Database: $DB_PATH"
echo ""

# Function to run write operations
run_writer() {
    local proc_id=$1
    for i in $(seq 1 $ITERATIONS); do
        python3 "$SCRIPTS_DIR/write_metrics.py" 2>&1 | grep -i "locked\|error" >> "$ERROR_LOG"
    done
}

# Launch concurrent processes
for i in $(seq 1 $NUM_PROCESSES); do
    run_writer $i &
done

# Wait for all background processes to complete
wait

# Count errors
ERROR_COUNT=$(grep -c "locked\|error" "$ERROR_LOG" 2>/dev/null || echo 0)

echo "Test completed."
echo "Processes tested: $NUM_PROCESSES"
echo "Errors encountered: $ERROR_COUNT"

if [ $ERROR_COUNT -eq 0 ]; then
    echo "Status: SUCCESS - No locking errors detected"
    exit 0
else
    echo "Status: FAILED - Database locking errors detected"
    exit 1
fi